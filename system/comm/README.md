# halo_comm

A small **ROS 2-style middleware over ZeroMQ (TCP) with protobuf messages**, used
for the HALO **tracker ⇄ generator** link. It is transport-only: it
moves typed messages between processes (and across machines) and nothing else —
the control logic lives in tracker / halo_system.

Two symmetric implementations share one wire format:

| | Source | Depends on |
|---|---|---|
| **Python** | [`python/halo_comm/node.py`](python/halo_comm/node.py) | `pyzmq`, `protobuf` |
| **C++** | [`include/halo_comm/node.hpp`](include/halo_comm/node.hpp) (header-only) | `libzmq` C API, `protobuf` |


---

## How it works

halo_comm wraps four ZeroMQ socket patterns behind a `Node` that owns the ZMQ
context and the endpoints you register on it:

| Primitive | ZMQ socket | Direction | Semantics |
|---|---|---|---|
| **Service** | `REP` | inbound request → reply | server side of a call; polled by `spin_once()` |
| **Client** | `REQ` | outbound request → reply | **blocking** call with timeouts; auto-rebuilds the socket on timeout (a timed-out `REQ` is otherwise stuck) |
| **Publisher** | `PUB` | one-way out | fire-and-forget topic (e.g. 50 Hz telemetry) |
| **Subscriber** | `SUB` | one-way in | freshest message via `recv()` poll, or a background `on_message` callback |

Key properties:

- **Roles, not peers.** A *server* `Node` calls `create_service` / `create_publisher`,
  then calls `spin_once()` from its loop (poll every `REP` socket and dispatch ready
  requests). A *client*
  `Node` calls `create_client` / `create_subscription` and uses the returned handles.
  A `Publisher` publishes; a `Client` calls — they are different socket types and
  are **not** interchangeable.
- **Conflate.** `PUB`/`SUB` default to `CONFLATE` (queue depth 1): a subscriber
  always sees the *latest* sample, never a backlog — right for 50 Hz state where
  stale frames are useless.
- **Protobuf on the wire.** The middleware only calls `SerializeToString()` /
  `ParseFromString()`. No hand byte-packing; the `.proto` schema is the contract.
- **TCP, cross-device.** Server binds `tcp://{bind_ip}:{port}`; client connects
  `tcp://{host}:{port}`. Set the client's host to the tracker's IP for a two-box
  setup, or `127.0.0.1` for localhost.
- **Clean teardown.** Every socket sets `LINGER=0` and is closed before
  `zmq_ctx_term` (otherwise context teardown blocks forever). In C++ the
  Publisher is move-only with a closing destructor; in Python
  call `Node.shutdown()`.

### The deployment channels

The tracker (server) exposes exactly these to the generator (client):

| Name | Pattern | Port | Messages | Purpose |
|---|---|---|---|---|
| `g1/get_state` | REQ/REP | 5555 | `StateRequest` → `RobotState` | pull robot state (+ optional ego `Image`) |
| `g1/put_plan`  | REQ/REP | 5556 | `MotionPlan` → `PlanAck`     | push a generated motion plan |
| `g1/get_trim`  | REQ/REP | 5557 | `TrimRequest` → `TrimReply`  | pull the tracker's remaining motion (RTC conditioning) |
| `g1/state`     | PUB/SUB | 5558 | `RobotState` (50 Hz)         | telemetry topic (image-free; generator, orchestrator, viewer) |
| `g1/obs`       | PUB/SUB | 5559 | `ObsImage` (~10 Hz)          | ego-image topic, **generator**-published (sim = the rendered frame it already pulled, real = its own camera); RAW or JPEG |

Channels are discovered at runtime through the master registry
(`HALO_COMM_REGISTRY`); the C++ side binds them in
`system/cpp/transport/tracker_server.hpp`.

---

## CLI — `halo_comm` (a `ros2 topic echo`-style tool)

A small command-line tool for **peeking at the wire** — subscribe, measure rate,
call a service, publish, or visualize the link. Installed as the `halo_comm` console
script by `pip install -e python`; also runs as `python -m halo_comm`.

```bash
halo_comm list                       # the channel registry (name, pattern, port, types)
halo_comm echo g1/state              # print every telemetry message live (ros2 topic echo)
halo_comm echo g1/state -n 5 --json  # first 5 messages as JSON  (--csv also available)
halo_comm echo g1/state --out state.log   # append messages to a file (bag-lite)
halo_comm hz g1/state                # measure the topic's receive rate (ros2 topic hz)
halo_comm call g1/get_state          # one-shot REQ/REP; prints the reply
halo_comm call g1/put_plan --file plan.csv     # request payload from an (n,36) CSV
halo_comm pub  g1/state -r 50        # publish (default: zero-valued RobotState) at 50 Hz
halo_comm graph                      # LIVE dashboard on http://127.0.0.1:8080, opens a browser
halo_comm monitor                    # LIVE channel status in the terminal (no browser, SSH-friendly)
```

Notes / flags:
- **Names, not ports.** ZMQ has no discovery (nothing on the wire carries a
  channel's name or type — only its TCP port), so the CLI resolves names through
  the master registry.
- **`echo`/`hz` see *every* message** (`conflate=False`); pass `--conflate` for
  freshest-only. A conflating subscriber would silently drop most of a 50 Hz stream.
- **Output** defaults to protobuf text (indented blocks); `--json` / `--csv` switch
  format, `--out FILE` appends. Large `image.rgb` bytes are elided unless `--full`.
- **`--host`** points at a remote tracker (default `127.0.0.1`); `call` takes
  `--timeout-ms`.
- **`graph`** runs a live dashboard (`http.server`, stdlib-only) on
  `http://127.0.0.1:8080` and opens a browser: the page polls `/api/graph` every
  ~1.5 s, so edges color up/down and topics show their measured Hz + latest sample
  in real time. `--no-open` skips the browser, `--port`/`--bind` move the listener,
  `--registry` picks the master registry. (Live-only — the page is worthless frozen.)
- **`monitor`** is the same live view in the terminal — a refreshing table of every
  channel's up/down + topic Hz + latest tick. No browser, no `:8080` bind; works
  over SSH. `-i/--interval` sets the refresh period.

---

## Layout

```
system/comm/
├── msgs/                     # message contracts — the source of truth
│   ├── state.proto           #   StateRequest, RobotState, RefMotion, Image, DepthImage, ObsImage
│   ├── motion.proto          #   MotionPlan, PlanAck, TrimRequest, TrimReply, MissionCommand, PlanStepMsg, MissionPlanMsg, GenStatus
│   └── discovery.proto       #   Endpoint, RegistryRequest/Reply
├── include/halo_comm/node.hpp   # C++ Node (header-only over libzmq)
├── python/
│   ├── halo_comm/               # Python package
│   │   ├── node.py           #   Node, Publisher, Subscriber, Client
│   │   ├── cli.py            #   `halo_comm` CLI: echo / list / hz / call / pub / graph / monitor
│   │   ├── graph.py          #   live dashboard server + HTML + LiveMonitor (graph & monitor)
│   │   ├── __main__.py       #   `python -m halo_comm`
│   │   └── msgs/*_pb2.py      #   generated (committed) — `make protos`
│   └── pyproject.toml        # `pip install -e system/comm/python` (+ `halo_comm` console script)
├── cmake/FindZeroMQ.cmake     # libzmq locator (libzmq ships no CMake config)
├── CMakeLists.txt            # builds `halo_comm::halo_comm`
└── Makefile                  # `make protoc29` / `make protos` / `make check-protos`
```

---

## Install / build

**Python** — installs `pyzmq` + `protobuf`; the generated `*_pb2.py` are committed,
so no `protoc` is needed on the host:

```bash
pip install -e system/comm/python # or: export PYTHONPATH=system/comm/python:$PYTHONPATH
```

**C++** — needs `protobuf` + `libzmq` (`brew install protobuf zeromq`). Consume it
from a parent CMake project:

```cmake
add_subdirectory(system/comm)
target_link_libraries(my_app PRIVATE halo_comm::halo_comm)
# halo_comm::halo_comm propagates the libzmq + protobuf includes AND the generated
# state.pb.h / motion.pb.h, so #include "state.pb.h" just works.
```

…or build it standalone:

```bash
cmake -S system/comm -B system/comm/build/cmake
cmake --build system/comm/build/cmake
```

---

## Creating or changing a message

1. **Edit the `.proto`** in [`msgs/`](msgs/). Rules (proto3):
   - **Never renumber or reuse a field tag** — the tag *is* the wire contract.
   - **Append** new fields with fresh tags; proto3 ignores unknown fields, so old
     readers stay compatible.
   - Mind the per-field conventions already in use: `RobotState` quaternions are
     **wxyz**, `MotionPlan`/`TrimReply` rows are **xyzw** — kept distinct on
     purpose, do not unify. Plan/trim rows are a flat `repeated double values`
     that is `cols` (= 36) wide, row-major: `[base_xyz(3), quat_xyzw(4), joints(29)]`.

   Example — add an optional field to `RobotState` in `msgs/state.proto`:

   ```protobuf
   message RobotState {
     repeated double joint_pos     = 1;
     repeated double imu_quat_wxyz = 2;
     repeated double base_pos      = 3;
     int32           tick          = 4;
     RefMotion       ref_motion    = 5;
     Image           image         = 6;
     double          battery_v     = 7;  // NEW: next free tag, appended
   }
   ```

2. **Regenerate the Python bindings — required whenever you change *any* message
   or type.** Editing a `.proto` has **no effect on its own**: the Python side
   imports the generated `*_pb2.py`, so you must rebuild them *and* reinstall the
   package before the new message/field/type is visible to any Python app.

   ```bash
   cd system/comm
   make protoc29                # one-time: fetch the pinned protoc 29 into .tooling/
   make protos                  # (re)generate python/halo_comm/msgs/*_pb2.py from msgs/*.proto
   pip install -e python        # reinstall so the new bindings/types are picked up
   ```

   `make protoc29` downloads the correct compiler for your OS/arch (see the
   toolchain note below for *why* the version is pinned); you only need it once.
   `make protos` then uses it automatically and **hard-fails** if it sees any
   other `protoc`, so you can't accidentally commit incompatible bindings.

   Then **commit the regenerated `*_pb2.py`** — they are checked in so consumers
   don't need `protoc` to `pip install`. (With an editable install, regenerated
   `*_pb2.py` are picked up on the next import without reinstalling; re-run
   `pip install -e python` anyway when you **add a new `.proto`** or installed
   non-editably — it is always safe.)

   The **C++** side needs no manual step — CMake's `protobuf_generate` recompiles
   `msgs/*.proto` into `state.pb.h` / `motion.pb.h` on every build.

3. **Keep both sides honest.** `make check-protos` regenerates and must produce
   no diff.

> **Codegen toolchain — use `protoc` release 29.x.** The generated `*_pb2.py`
> carry a gencode version stamp, and at import time `google.protobuf` enforces
> **runtime ≥ gencode** (same major). This project pins the runtime to
> **protobuf 5.29.x** (`python/pyproject.toml`: `protobuf>=5.29,<6`), so you must
> generate with **protoc release 29.x**, which stamps gencode `5.29.x`. A newer
> protoc (e.g. Homebrew's `protoc 35`) stamps gencode `7.x`, and a 5.29 runtime
> then refuses to load it:
> `VersionError: ... gencode 7.34.0 runtime 5.29.6. Runtime version cannot be
> older than the linked gencode version`.
>
> **Just run `make protoc29`.** It downloads the pinned protoc 29 for your
> OS/arch (macOS arm64/x86_64, Linux x86_64/aarch64) into `system/comm/.tooling/`
> (git-ignored), needs no Homebrew and touches no `PATH`. `make protos` then
> picks it up automatically and **hard-fails** on any other `protoc`, so broken
> gencode can't be generated or committed:
>
> ```bash
> cd system/comm
> make protoc29                                 # one-time, -> libprotoc 29.0
> make protos
> grep -A4 ValidateProtobufRuntimeVersion python/halo_comm/msgs/state_pb2.py   # expect: 5, 29, 0
> ```
>
> This pin is for the **committed Python `*_pb2.py` only** — the C++ build
> regenerates `state.pb.h`/`motion.pb.h` against whatever `libprotobuf` it links
> (both from the same install), so it is self-consistent at any protoc version.

---

## Using it in Python

```python
from halo_comm import Node, state_pb2, motion_pb2
```

### Server (the tracker)

```python
node = Node("tracker")

# A service handler takes the parsed request and returns the reply message.
def on_get_state(req: state_pb2.StateRequest) -> state_pb2.RobotState:
    return state_pb2.RobotState(
        joint_pos=read_joints(),            # 29
        imu_quat_wxyz=[1, 0, 0, 0],
        base_pos=[0.0, 0.0, 0.75],
        tick=loop_counter,
    )

node.create_service(state_pb2.StateRequest, state_pb2.RobotState,
                    "g1/get_state", port=5555, handler=on_get_state)

state_pub = node.create_publisher(state_pb2.RobotState, "g1/state", port=5558)

# Drive services + the 50 Hz topic from one loop:
while running:
    node.spin_once(timeout_ms=0)            # dispatch any pending requests
    state_pub.publish(current_state())      # fire-and-forget telemetry
    sleep_to_50hz()
node.shutdown()                             # closes every socket
```

### Client (the generator)

```python
node = Node("generator")

state_cli = node.create_client(state_pb2.StateRequest, state_pb2.RobotState,
                               "g1/get_state", host="10.0.0.5", port=5555)
plan_cli  = node.create_client(motion_pb2.MotionPlan, motion_pb2.PlanAck,
                               "g1/put_plan", host="10.0.0.5", port=5556)

# Blocking REQ/REP. Raises zmq.Again on timeout (socket is auto-rebuilt so the
# next call retries cleanly).
try:
    state = state_cli.call(state_pb2.StateRequest())
except zmq.Again:
    ...  # tracker not responding

ack = plan_cli.call(motion_pb2.MotionPlan(input_fps=30, cols=36, values=flat_rows))

# Subscribe to telemetry — poll for the freshest sample…
sub = node.create_subscription(state_pb2.RobotState, "g1/state",
                               host="10.0.0.5", port=5558, conflate=True)
latest = sub.recv(timeout_ms=20)            # RobotState or None

# …or receive via a background callback thread instead:
sub = node.create_subscription(state_pb2.RobotState, "g1/state",
                               host="10.0.0.5", port=5558,
                               on_message=lambda m: record(m))

node.shutdown()
```

---

## Using it in C++

```cpp
#include "halo_comm/node.hpp"
#include "state.pb.h"
#include "motion.pb.h"
using namespace motion_msgs;
```

### Server (the tracker)

```cpp
halo_comm::Node node("tracker");

// Handler signature is (const Req&, Rep& out) — fill `out` in place.
node.create_service<StateRequest, RobotState>("g1/get_state", 5555,
    [&](const StateRequest&, RobotState& out) {
        for (double q : read_joints()) out.add_joint_pos(q);   // 29
        for (double q : {1.0, 0.0, 0.0, 0.0}) out.add_imu_quat_wxyz(q);
        out.set_tick(loop_counter);
    });

auto state_pub = node.create_publisher<RobotState>("g1/state", 5558);

while (running) {
    node.spin_once(/*timeout_ms=*/0);   // dispatch any pending requests
    state_pub.publish(current_state());
    sleep_to_50hz();
}                                   // sockets close on scope exit
```

(`node.spin_once(timeout_ms)` polls the `REP` sockets exactly once; the caller drives the
loop.)
