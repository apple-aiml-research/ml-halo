//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// halo_comm discovery — C++ mirror of python/halo_comm/discovery.py.
//
// Master registry (ROS2 "Discovery Server" model): providers that BIND a
// service/publisher register their real host:port with one configured registry
// (HALO_COMM_REGISTRY=host:port), then heartbeat; consumers query it. Header-only,
// self-contained (raw libzmq C API + discovery.pb.h) so it does NOT include
// node.hpp — node.hpp includes THIS. Everything is best-effort: discovery must
// never crash or block a node. HALO_COMM_REGISTRY picks the registry (default localhost).

#include <zmq.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "discovery.pb.h"

namespace halo_comm::discovery {

constexpr int         kDefaultRegistryPort = 5560;
constexpr double      kDefaultTtlS         = 10.0;
inline const char*    kDefaultRegistryHost = "127.0.0.1";  // export HALO_COMM_REGISTRY for multi-host

struct Addr {
    std::string host;
    int         port{kDefaultRegistryPort};
};

// Master registry address. Discovery is ALWAYS ON: HALO_COMM_REGISTRY=host:port, or
// the localhost default. For multi-host, export HALO_COMM_REGISTRY on every machine.
inline Addr registry_addr() {
    const char* v = std::getenv("HALO_COMM_REGISTRY");
    if (v == nullptr || *v == '\0') return Addr{kDefaultRegistryHost, kDefaultRegistryPort};
    std::string s(v);
    const auto pos = s.rfind(':');
    if (pos == std::string::npos) return Addr{s, kDefaultRegistryPort};
    try {
        return Addr{s.substr(0, pos), std::stoi(s.substr(pos + 1))};
    } catch (...) {
        return Addr{s, kDefaultRegistryPort};
    }
}

// Local IP consumers can reach this node at: the source IP on the route toward
// the registry (UDP "connect" sets the route, sends nothing). HALO_COMM_ADVERTISE_HOST
// overrides. Falls back to 127.0.0.1.
inline std::string advertise_host(const std::string& registry_host) {
    if (const char* e = std::getenv("HALO_COMM_ADVERTISE_HOST"); e != nullptr && *e != '\0') {
        return e;
    }
    std::string result = "127.0.0.1";
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return result;
    // Resolve the registry host (accepts IP or hostname).
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(registry_host.c_str(), "9", &hints, &res) == 0 && res != nullptr) {
        if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
            sockaddr_in local{};
            socklen_t   len = sizeof(local);
            if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &len) == 0) {
                char buf[INET_ADDRSTRLEN] = {0};
                if (::inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf)) != nullptr) {
                    result = buf;
                }
            }
        }
        ::freeaddrinfo(res);
    }
    ::close(fd);
    return result;
}

inline bool is_local_host(const std::string& host) {
    if (host == "127.0.0.1" || host == "localhost" || host == "::1" || host == "0.0.0.0") {
        return true;
    }
    return advertise_host(host) == host;   // route toward host lands on host => local iface
}

// Non-blocking TCP connect with a short timeout — "is something bound there?"
inline bool port_listening(const std::string& host, int port, int timeout_ms = 250) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 ||
        res == nullptr) {
        return false;
    }
    const int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
        ::freeaddrinfo(res);
        return false;
    }
    ::fcntl(fd, F_SETFL, O_NONBLOCK);
    bool listening = false;
    const int rc = ::connect(fd, res->ai_addr, res->ai_addrlen);
    if (rc == 0) {
        listening = true;
    } else {
        fd_set wset;
        FD_ZERO(&wset);
        FD_SET(fd, &wset);
        timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        if (::select(fd + 1, nullptr, &wset, nullptr, &tv) > 0) {
            int       err = 0;
            socklen_t len = sizeof(err);
            ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
            listening = (err == 0);
        }
    }
    ::close(fd);
    ::freeaddrinfo(res);
    return listening;
}

// One best-effort REQ/REP to the registry (transient socket; short timeouts so a
// down registry never stalls a node). Returns true if a reply was parsed.
inline bool call_registry(void* ctx, const Addr& addr,
                          const motion_msgs::RegistryRequest& req,
                          motion_msgs::RegistryReply* rep) {
    void* sock = zmq_socket(ctx, ZMQ_REQ);
    if (sock == nullptr) return false;
    const int rcv = 400, snd = 400, linger = 0;
    zmq_setsockopt(sock, ZMQ_RCVTIMEO, &rcv, sizeof(rcv));
    zmq_setsockopt(sock, ZMQ_SNDTIMEO, &snd, sizeof(snd));
    zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
    const std::string a = "tcp://" + addr.host + ":" + std::to_string(addr.port);
    bool ok = false;
    if (zmq_connect(sock, a.c_str()) == 0) {
        std::string buf;
        if (req.SerializeToString(&buf) &&
            zmq_send(sock, buf.data(), buf.size(), 0) >= 0) {
            zmq_msg_t m;
            zmq_msg_init(&m);
            if (zmq_msg_recv(&m, sock, 0) >= 0 && rep != nullptr) {
                ok = rep->ParseFromArray(zmq_msg_data(&m),
                                         static_cast<int>(zmq_msg_size(&m)));
            }
            zmq_msg_close(&m);
        }
    }
    zmq_close(sock);
    return ok;
}

// ── Embedded registry server (C++ port of python/halo_comm/registry.py) ─────────
// So a node whose HALO_COMM_REGISTRY names ITSELF can host the master registry
// IN-PROCESS, with no dependency on the Python `halo_comm` CLI being on PATH (the
// old std::system("halo_comm registry") auto-spawn). Raw libzmq only — this header
// is included BY node.hpp, so it must not use Node.

inline double now_s() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Endpoint table with per-entry TTL expiry (mirror of discovery.py RegistryTable).
// Single-threaded: owned + accessed only by the serve loop below.
class RegistryTable {
public:
    explicit RegistryTable(double default_ttl_s = kDefaultTtlS) : ttl_(default_ttl_s) {}

    void put(const motion_msgs::Endpoint& ep, double ttl_s, double now) {
        const double ttl = (ttl_s > 0.0) ? ttl_s : ttl_;
        motion_msgs::Endpoint e = ep;
        e.set_stamp(now);
        eps_[key(ep)] = Entry{e, now + ttl};
    }
    void erase(const motion_msgs::Endpoint& ep) { eps_.erase(key(ep)); }
    int prune(double now) {
        int n = 0;
        for (auto it = eps_.begin(); it != eps_.end();) {
            if (it->second.expire <= now) { it = eps_.erase(it); ++n; }
            else { ++it; }
        }
        return n;
    }
    std::vector<motion_msgs::Endpoint> list(double now) {
        prune(now);
        std::vector<motion_msgs::Endpoint> out;
        out.reserve(eps_.size());
        for (const auto& kv : eps_) out.push_back(kv.second.ep);
        return out;
    }
    std::size_t size() const { return eps_.size(); }

private:
    // Providers differ by (name,host,port); consumers (port 0) differ by node+pid,
    // so many consumers of one channel coexist. Matches discovery.py _key().
    using Key = std::tuple<std::string, std::string, int, std::string, int>;
    static Key key(const motion_msgs::Endpoint& e) {
        return {e.name(), e.host(), e.port(), e.node(), e.pid()};
    }
    struct Entry { motion_msgs::Endpoint ep; double expire; };
    std::map<Key, Entry> eps_;
    double ttl_;
};

// Pure request handler (mirror of registry.py handle()).
inline void registry_handle(RegistryTable& table,
                            const motion_msgs::RegistryRequest& req,
                            motion_msgs::RegistryReply* rep) {
    using R = motion_msgs::RegistryRequest;
    const double now = now_s();
    rep->set_ok(true);
    rep->set_now(now);
    switch (req.op()) {
        case R::REGISTER:   table.put(req.endpoint(), req.ttl_s(), now); break;
        case R::UNREGISTER: table.erase(req.endpoint()); break;
        case R::LIST:
            for (const auto& ep : table.list(now)) *rep->add_endpoints() = ep;
            break;
        default: rep->set_ok(false); break;
    }
}

// Blocking REP serve loop on tcp://0.0.0.0:port. Returns immediately (does
// nothing) if the port is already bound by another registry — idempotent, like
// the Python detached-spawn. Runs until *running goes false.
inline void registry_serve_loop(void* ctx, int port, std::atomic<bool>* running) {
    void* sock = zmq_socket(ctx, ZMQ_REP);
    if (sock == nullptr) return;
    const int rcvtimeo = 200, linger = 0;
    zmq_setsockopt(sock, ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
    zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
    const std::string a = "tcp://0.0.0.0:" + std::to_string(port);
    if (zmq_bind(sock, a.c_str()) != 0) {
        // Someone else already owns the port (another registry) — not our job.
        zmq_close(sock);
        return;
    }
    std::fprintf(stderr, "[halo_comm] embedded registry serving on %s\n", a.c_str());
    RegistryTable table;
    double last_prune = now_s();
    while (running->load(std::memory_order_acquire)) {
        zmq_msg_t m;
        zmq_msg_init(&m);
        const int n = zmq_msg_recv(&m, sock, 0);
        if (n >= 0) {
            motion_msgs::RegistryRequest req;
            motion_msgs::RegistryReply rep;
            if (req.ParseFromArray(zmq_msg_data(&m), static_cast<int>(zmq_msg_size(&m)))) {
                registry_handle(table, req, &rep);
            } else {
                rep.set_ok(false);
                rep.set_now(now_s());
            }
            zmq_msg_close(&m);
            std::string buf;
            if (rep.SerializeToString(&buf)) {
                zmq_send(sock, buf.data(), buf.size(), 0);  // REP must reply
            }
        } else {
            zmq_msg_close(&m);   // RCVTIMEO — re-check running and continue
        }
        const double t = now_s();
        if (t - last_prune >= 2.0) { table.prune(t); last_prune = t; }
    }
    zmq_close(sock);
}

// Start the embedded registry ONCE, on a detached background thread with its own
// ZMQ context (independent of any Node). Idempotent: redundant calls no-op. The
// thread + context intentionally live for the process lifetime (a daemon).
inline void start_embedded_registry(int port) {
    static std::atomic<bool> started{false};
    bool expected = false;
    if (!started.compare_exchange_strong(expected, true)) return;   // once
    static std::atomic<bool> running{true};
    static void* ctx = zmq_ctx_new();
    if (ctx == nullptr) return;
    std::thread([port]() { registry_serve_loop(ctx, port, &running); }).detach();
}

// Bootstrap the master registry. If HALO_COMM_REGISTRY names THIS machine and
// nothing is listening, start the embedded registry (idempotent). No-op for a
// remote master. Returns true if a registry is available locally.
inline bool init() {
    const Addr addr = registry_addr();
    if (!is_local_host(addr.host)) {
        // Remote master: this node does not own the registry — the daemon must
        // already be running on addr.host. We only register/heartbeat to it.
        std::fprintf(stderr,
                     "[halo_comm] registry %s:%d is REMOTE — not auto-starting here; "
                     "ensure a registry daemon is running on that host.\n",
                     addr.host.c_str(), addr.port);
        return false;
    }
    if (port_listening(addr.host, addr.port)) {
        std::fprintf(stderr, "[halo_comm] registry already up at %s:%d — using it.\n",
                     addr.host.c_str(), addr.port);
        return true;
    }
    std::fprintf(stderr,
                 "[halo_comm] no registry at %s:%d — starting embedded registry...\n",
                 addr.host.c_str(), addr.port);
    start_embedded_registry(addr.port);                // in-process; no halo_comm CLI needed
    for (int i = 0; i < 40; ++i) {
        if (port_listening(addr.host, addr.port)) {
            std::fprintf(stderr, "[halo_comm] embedded registry OK — listening on %s:%d.\n",
                         addr.host.c_str(), addr.port);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::fprintf(stderr,
                 "[halo_comm] embedded registry did NOT come up on %s:%d after 2s "
                 "(port taken, or bind failed). Nodes will keep waiting for the registry.\n",
                 addr.host.c_str(), addr.port);
    return false;
}

// Registers a node's bound endpoints with the master registry and heartbeats to
// keep them alive; unregisters on shutdown. Owned by a Node.
class Registrar {
public:
    explicit Registrar(void* ctx, std::string node) : ctx_(ctx), node_(std::move(node)) {
        addr_      = registry_addr();               // always-on (localhost default)
        advertise_ = advertise_host(addr_.host);
    }
    ~Registrar() { shutdown(); }
    Registrar(const Registrar&)            = delete;
    Registrar& operator=(const Registrar&) = delete;

    // Called by Node::create_service / create_publisher after the bind.
    void announce(const std::string& name, const std::string& pattern, int port,
                  const std::string& req_type = "", const std::string& rep_type = "",
                  const std::string& msg_type = "") {
        if (!enabled_) return;
        motion_msgs::Endpoint ep;
        ep.set_name(name);
        ep.set_pattern(pattern);
        ep.set_host(advertise_);
        ep.set_port(port);
        ep.set_node(node_);
        ep.set_req_type(req_type);
        ep.set_rep_type(rep_type);
        ep.set_msg_type(msg_type);
        ep.set_pid(static_cast<int>(::getpid()));
        {
            std::lock_guard<std::mutex> lk(mu_);
            eps_.push_back(ep);
        }
        send(motion_msgs::RegistryRequest::REGISTER, ep);
        std::fprintf(stderr, "[halo_comm] registered %s @ %s:%d -> registry %s:%d\n",
                     name.c_str(), advertise_.c_str(), port,
                     addr_.host.c_str(), addr_.port);
        start_heartbeat();
    }

    void shutdown() {
        if (!enabled_) return;
        running_.store(false);
        if (hb_.joinable()) hb_.join();
        std::vector<motion_msgs::Endpoint> eps;
        {
            std::lock_guard<std::mutex> lk(mu_);
            eps.swap(eps_);
        }
        for (const auto& ep : eps) send(motion_msgs::RegistryRequest::UNREGISTER, ep);
        enabled_ = false;
    }

private:
    void send(motion_msgs::RegistryRequest::Op op, const motion_msgs::Endpoint& ep) {
        motion_msgs::RegistryRequest req;
        req.set_op(op);
        *req.mutable_endpoint() = ep;
        motion_msgs::RegistryReply rep;
        call_registry(ctx_, addr_, req, &rep);          // best-effort
    }

    void start_heartbeat() {
        bool expected = false;
        if (!started_.compare_exchange_strong(expected, true)) return;   // once
        running_.store(true);
        hb_ = std::thread([this]() {
            const auto period = std::chrono::milliseconds(
                static_cast<int>(kDefaultTtlS / 3.0 * 1000));
            while (running_.load()) {
                std::this_thread::sleep_for(period);
                if (!running_.load()) break;
                std::vector<motion_msgs::Endpoint> eps;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    eps = eps_;
                }
                for (const auto& ep : eps) send(motion_msgs::RegistryRequest::REGISTER, ep);
            }
        });
    }

    void*                               ctx_{nullptr};
    std::string                         node_;
    Addr                                addr_;
    std::string                         advertise_;
    bool                                enabled_{true};   // false after shutdown()
    std::vector<motion_msgs::Endpoint>  eps_;
    std::mutex                          mu_;
    std::thread                         hb_;
    std::atomic<bool>                   running_{false};
    std::atomic<bool>                   started_{false};
};

}  // namespace halo_comm::discovery
