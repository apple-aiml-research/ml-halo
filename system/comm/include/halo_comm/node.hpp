//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// halo_comm — ROS2-style middleware over ZMQ (TCP) with protobuf messages.
// C++ mirror of python/halo_comm/node.py. Header-only; uses the libzmq C API
// (no cppzmq dependency) + protobuf-generated message types.
//
//   halo_comm::Node node("tracker");
//   node.create_service<StateRequest, RobotState>("g1/get_state", 5555,
//       [&](const StateRequest&, RobotState& out){ fill_state(out); });
//   auto pub = node.create_publisher<RobotState>("g1/state", 5558);
//   node.spin_once();                      // call from your loop to dispatch REP requests
//   pub.publish(state_msg);                // 50 Hz, one-way
//
// Server side only (the tracker serves and publishes; clients are Python).
// Serialize/parse is protobuf; no hand byte-packing. Transport is TCP
// (cross-device): the server binds tcp://{bind_ip}:{port}.

#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "halo_comm/discovery.hpp"   // master-registry auto-register

namespace halo_comm {

// Bootstrap the master registry if this machine is the master and none is
// running. No-op if the master is remote. Call once
// at process startup (e.g. in a tracker main()).
inline bool init() { return discovery::init(); }

inline std::string addr(const std::string& host, int port) {
    return "tcp://" + host + ":" + std::to_string(port);
}

// Bind, retrying for up to 5 s (a previous run may still be releasing the port).
// A socket that never binds is reported, not silently left deaf.
inline bool bind_or_report(void* sock, const std::string& a, const std::string& name) {
    int first_err = 0;
    for (int i = 0; i < 50; ++i) {
        if (zmq_bind(sock, a.c_str()) == 0) return true;
        if (first_err == 0) first_err = zmq_errno();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::fprintf(stderr, "[halo_comm] ERROR: %s could not bind %s: %s (errno %d)\n",
                 name.c_str(), a.c_str(), zmq_strerror(first_err), first_err);
    return false;
}

// ── Publisher (PUB) ──────────────────────────────────────────────────────────
template <typename Msg>
class Publisher {
public:
    Publisher(void* ctx, const std::string& name, int port,
              const std::string& bind_ip = "0.0.0.0") {
        sock_ = zmq_socket(ctx, ZMQ_PUB);
        const int sndhwm = 1;
        zmq_setsockopt(sock_, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
        const int linger = 0;
        zmq_setsockopt(sock_, ZMQ_LINGER, &linger, sizeof(linger));
        const std::string a = addr(bind_ip, port);
        bind_or_report(sock_, a, name);
    }
    void publish(const Msg& msg) {
        (void)msg.SerializeToString(&buf_);
        zmq_send(sock_, buf_.data(), buf_.size(), ZMQ_DONTWAIT);
    }

    // Move-only + closing dtor: a socket must be closed before zmq_ctx_term,
    // else ctx teardown blocks forever. Moved-from handles null their socket.
    ~Publisher() { if (sock_) zmq_close(sock_); }
    Publisher(Publisher&& o) noexcept
        : sock_(std::exchange(o.sock_, nullptr)), buf_(std::move(o.buf_)) {}
    Publisher& operator=(Publisher&& o) noexcept {
        if (this != &o) {
            if (sock_) zmq_close(sock_);
            sock_ = std::exchange(o.sock_, nullptr);
            buf_  = std::move(o.buf_);
        }
        return *this;
    }
    Publisher(const Publisher&)            = delete;
    Publisher& operator=(const Publisher&) = delete;

private:
    void*       sock_{nullptr};
    std::string buf_;
};

// ── Service (REP) — type-erased base + typed impl for the poller ─────────────
class ServiceBase {
public:
    // Close the REP socket on destruction — must happen before zmq_ctx_term
    // (the Node clears services_ in its dtor before terminating the context).
    virtual ~ServiceBase() { if (sock_) zmq_close(sock_); }
    virtual void handle_ready() = 0;
    [[nodiscard]] void* socket() const noexcept { return sock_; }

protected:
    void* sock_{nullptr};
};

template <typename Req, typename Rep, typename Handler>
class Service final : public ServiceBase {
public:
    Service(void* ctx, const std::string& name, int port, Handler handler,
            const std::string& bind_ip)
        : handler_(std::move(handler)) {
        sock_ = zmq_socket(ctx, ZMQ_REP);
        const std::string a = addr(bind_ip, port);
        bind_or_report(sock_, a, name);
    }
    void handle_ready() override {
        zmq_msg_t m;
        zmq_msg_init(&m);
        if (zmq_msg_recv(&m, sock_, 0) < 0) { zmq_msg_close(&m); return; }
        Req req;
        (void)req.ParseFromArray(zmq_msg_data(&m), static_cast<int>(zmq_msg_size(&m)));
        zmq_msg_close(&m);
        Rep rep;
        handler_(req, rep);                    // handler(const Req&, Rep& out)
        std::string buf;
        (void)rep.SerializeToString(&buf);
        zmq_send(sock_, buf.data(), buf.size(), 0);
    }

private:
    Handler     handler_;
};

// ── Node ─────────────────────────────────────────────────────────────────────
class Node {
public:
    explicit Node(std::string name)
        : name_(std::move(name)), ctx_(zmq_ctx_new()), registrar_(ctx_, name_) {}
    // Close all owned sockets (the service REP sockets) BEFORE terminating the
    // context, otherwise zmq_ctx_term blocks forever waiting on open sockets.
    // Caller-held Publisher handles must be destroyed before this Node (scope/member
    // order ensures that); each closes its own socket in its dtor.
    ~Node() {
        registrar_.shutdown();       // join heartbeat + unregister BEFORE ctx_term
        services_.clear();           // ServiceBase dtors close the REP sockets
        if (ctx_) zmq_ctx_term(ctx_);
    }
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    template <typename Msg>
    Publisher<Msg> create_publisher(const std::string& name, int port,
                                    const std::string& bind_ip = "0.0.0.0") {
        Publisher<Msg> pub(ctx_, name, port, bind_ip);
        registrar_.announce(name, "PUB/SUB", port, /*req*/"", /*rep*/"",
                            std::string(Msg::descriptor()->name()));   // auto-register
        return pub;
    }
    template <typename Req, typename Rep, typename Handler>
    void create_service(const std::string& name, int port, Handler handler,
                        const std::string& bind_ip = "0.0.0.0") {
        services_.push_back(std::make_unique<Service<Req, Rep, Handler>>(
            ctx_, name, port, std::move(handler), bind_ip));
        registrar_.announce(name, "REQ/REP", port,
                            std::string(Req::descriptor()->name()),
                            std::string(Rep::descriptor()->name()));   // auto-register
    }

    // Poll all REP sockets once and dispatch ready requests.
    void spin_once(int timeout_ms = 100) {
        if (services_.empty()) return;
        std::vector<zmq_pollitem_t> items;
        items.reserve(services_.size());
        for (auto& s : services_) items.push_back({s->socket(), 0, ZMQ_POLLIN, 0});
        if (zmq_poll(items.data(), static_cast<int>(items.size()), timeout_ms) <= 0) return;
        for (std::size_t i = 0; i < services_.size(); ++i) {
            if ((items[i].revents & ZMQ_POLLIN) != 0) services_[i]->handle_ready();
        }
    }

private:
    std::string                                name_;
    void*                                      ctx_{nullptr};
    discovery::Registrar                       registrar_;   // auto-register
    std::vector<std::unique_ptr<ServiceBase>>  services_;
};

}  // namespace halo_comm
