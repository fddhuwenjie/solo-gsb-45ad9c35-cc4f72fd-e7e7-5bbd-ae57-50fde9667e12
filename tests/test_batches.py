"""现场设备批次链接口测试：链式追加、摘要、幂等重放、冲突与并发。"""
import hashlib
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.models import BatchUploadRequest

THRESHOLDS = {"tolerance": 0.5, "max_drift": 5e-4, "max_missed": 3, "max_false": 3}
DEVICE = "dev-01"


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def obs(n, base=1_700_000_000.0, step=0.5):
    return [
        {"local_id": f"o{i:03d}", "captured_at": base + i * step}
        for i in range(n)
    ]


def batch(batch_id, sequence, previous_digest="", observations=None, **kw):
    body = {
        "batch_id": batch_id,
        "sequence": sequence,
        "previous_digest": previous_digest,
        "observations": observations if observations is not None else obs(3),
        **THRESHOLDS,
    }
    body.update(kw)
    return body


def expected_digest(device_id, body):
    """按约定规则独立复算批次摘要：排除摘要字段，UTF-8 键序紧凑 JSON 的 SHA-256。"""
    payload = {
        "batch_id": body["batch_id"],
        "device_id": device_id,
        "max_drift": body["max_drift"],
        "max_false": body["max_false"],
        "max_missed": body["max_missed"],
        "observations": [
            {"captured_at": o["captured_at"], "local_id": o["local_id"]}
            for o in body["observations"]
        ],
        "sequence": body["sequence"],
        "tolerance": body["tolerance"],
    }
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def upload(client, body, device=DEVICE):
    return client.post(f"/field-devices/{device}/batches", json=body)


def parse_ts(s):
    """查询接口返回的 ISO 时间戳可解析（含时区）。"""
    return datetime.fromisoformat(s)


# ---------------- 链式追加与摘要 ----------------


def test_first_batch_accepted_and_digest(client):
    body = batch("b-1", 1)
    r = upload(client, body)
    assert r.status_code == 201
    resp = r.json()
    assert resp["status"] == "accepted"
    assert resp["device_id"] == DEVICE
    assert resp["batch_id"] == "b-1" and resp["sequence"] == 1
    assert resp["previous_digest"] == ""
    assert resp["observation_count"] == 3
    # 摘要与服务端约定一致：排除 previous_digest，UTF-8 键序紧凑 JSON 的 SHA-256
    assert resp["digest"] == expected_digest(DEVICE, body)
    assert resp["chain"] == {
        "head_sequence": 1,
        "head_digest": resp["digest"],
        "next_sequence": 2,
        "expected_previous_digest": resp["digest"],
    }

    # 游标反映链头
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 1
    assert cur["head_digest"] == resp["digest"]
    assert cur["next_sequence"] == 2
    assert cur["expected_previous_digest"] == resp["digest"]
    assert cur["batch_count"] == 1


def test_first_batch_requires_empty_previous_digest(client):
    body = batch("b-1", 1, previous_digest="ab" * 32)
    r = upload(client, body)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "stale_chain_head"
    assert detail["expected_sequence"] == 1
    assert detail["expected_previous_digest"] == ""


def test_chain_append_and_stale_head(client):
    r1 = upload(client, batch("b-1", 1))
    d1 = r1.json()["digest"]

    # 链头过期：previous_digest 不是当前链头
    r = upload(client, batch("b-2", 2, previous_digest="00" * 32))
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "stale_chain_head"
    assert detail["expected_sequence"] == 2
    assert detail["expected_previous_digest"] == d1

    # 正确引用链头 → 接受
    r2 = upload(client, batch("b-2", 2, previous_digest=d1))
    assert r2.status_code == 201
    assert r2.json()["previous_digest"] == d1

    # 再次用旧链头 → 409，期望值已前移
    r = upload(client, batch("b-3", 3, previous_digest=d1))
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["expected_sequence"] == 3
    assert detail["expected_previous_digest"] == r2.json()["digest"]


def test_sequence_must_extend_head(client):
    upload(client, batch("b-1", 1))
    d1 = client.get(f"/field-devices/{DEVICE}/cursor").json()["head_digest"]
    # 跳号：sequence 不是链头 + 1
    r = upload(client, batch("b-3", 3, previous_digest=d1))
    assert r.status_code == 409
    assert r.json()["detail"]["expected_sequence"] == 2


def test_digest_covers_payload_and_device(client):
    # 同一载荷在不同设备上摘要不同（device_id 参与摘要）
    body = batch("b-1", 1)
    r_a = upload(client, body, device="dev-a")
    r_b = upload(client, body, device="dev-b")
    assert r_a.status_code == 201 and r_b.status_code == 201
    assert r_a.json()["digest"] != r_b.json()["digest"]
    assert r_a.json()["digest"] == expected_digest("dev-a", body)

    # 非 ASCII 内容按 UTF-8 参与摘要（ensure_ascii=False）
    body2 = batch("b-2", 1, observations=[
        {"local_id": "浮标-甲", "captured_at": 1_700_000_000.0},
        {"local_id": "浮标-乙", "captured_at": 1_700_000_001.0},
    ])
    r = upload(client, body2, device="dev-c")
    assert r.status_code == 201
    assert r.json()["digest"] == expected_digest("dev-c", body2)


# ---------------- 幂等重放与冲突 ----------------


def test_replay_same_payload_returns_original(client):
    body = batch("b-1", 1)
    r1 = upload(client, body)
    assert r1.status_code == 201
    original = r1.json()

    r2 = upload(client, body)
    assert r2.status_code == 200
    replayed = r2.json()
    assert replayed["status"] == "replayed"
    # 原结果内容不变（仅状态标记为 replayed）
    assert {k: v for k, v in replayed.items() if k != "status"} == {
        k: v for k, v in original.items() if k != "status"
    }

    # 链头与记录数不变
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 1 and cur["batch_count"] == 1
    assert client.get(f"/field-devices/{DEVICE}/batches").json()["count"] == 1


def test_same_batch_id_different_payload_conflict(client):
    r1 = upload(client, batch("b-1", 1))
    assert r1.status_code == 201
    d1 = r1.json()["digest"]

    # 同 batch_id、不同观测 → 409，附既有批次与期望值
    r = upload(client, batch("b-1", 2, previous_digest=d1, observations=obs(4)))
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "payload_changed"
    assert detail["existing_batch"] == {
        "sequence": 1, "batch_id": "b-1", "digest": d1,
    }
    assert detail["expected_sequence"] == 2
    assert detail["expected_previous_digest"] == d1

    # 同 batch_id、不同序号 → 409
    r = upload(client, batch("b-1", 2, previous_digest=d1))
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "payload_changed"


def test_same_sequence_different_payload_conflict(client):
    r1 = upload(client, batch("b-1", 1))
    assert r1.status_code == 201

    # 同 sequence、不同 batch_id → 409
    r = upload(client, batch("b-x", 1))
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "payload_changed"
    assert detail["existing_batch"]["batch_id"] == "b-1"
    assert detail["existing_batch"]["sequence"] == 1


def test_replay_after_chain_advanced(client):
    """链头前移后重放历史批次：相同载荷仍返回原结果，不按链头过期处理。"""
    r1 = upload(client, batch("b-1", 1))
    d1 = r1.json()["digest"]
    upload(client, batch("b-2", 2, previous_digest=d1))

    r = upload(client, batch("b-1", 1))
    assert r.status_code == 200
    assert r.json()["status"] == "replayed"
    assert r.json()["digest"] == d1


# ---------------- 请求校验 ----------------


def test_validation_errors(client):
    # 观测为空 / 超过 100 条
    assert upload(client, batch("b-1", 1, observations=[])).status_code == 422
    assert upload(client, batch("b-1", 1, observations=obs(101))).status_code == 422

    # 未按 (captured_at, local_id) 递增
    unsorted = [
        {"local_id": "a", "captured_at": 2.0},
        {"local_id": "b", "captured_at": 1.0},
    ]
    assert upload(client, batch("b-1", 1, observations=unsorted)).status_code == 422
    # captured_at 相同但 local_id 逆序
    tie_unsorted = [
        {"local_id": "b", "captured_at": 1.0},
        {"local_id": "a", "captured_at": 1.0},
    ]
    assert upload(client, batch("b-1", 1, observations=tie_unsorted)).status_code == 422
    # (captured_at, local_id) 完全重复
    dup = [
        {"local_id": "a", "captured_at": 1.0},
        {"local_id": "a", "captured_at": 1.0},
    ]
    assert upload(client, batch("b-1", 1, observations=dup)).status_code == 422
    # captured_at 相同、local_id 递增 → 合法
    tie_ok = [
        {"local_id": "a", "captured_at": 1.0},
        {"local_id": "b", "captured_at": 1.0},
    ]
    assert upload(client, batch("b-1", 1, observations=tie_ok)).status_code == 201

    # previous_digest 格式非法 / sequence 非法 / 阈值非法
    assert upload(client, batch("b-2", 1, previous_digest="xyz")).status_code == 422
    assert upload(client, batch("b-2", 0)).status_code == 422
    assert upload(client, batch("b-2", 1, tolerance=0)).status_code == 422
    assert upload(client, batch("b-2", 1, max_drift=1.0)).status_code == 422
    assert upload(client, batch("b-2", 1, max_missed=-1)).status_code == 422
    assert upload(client, batch("b-2", 1, max_false=-1)).status_code == 422

    # captured_at 非有限值：模型层拒绝（NaN/Inf 经 HTTP 提交时与原 API 一致，
    # 由 FastAPI 默认校验错误处理返回，不在本接口新增特殊行为）
    with pytest.raises(ValidationError):
        BatchUploadRequest(
            batch_id="b-2",
            sequence=1,
            observations=[{"local_id": "a", "captured_at": float("nan")}],
            **THRESHOLDS,
        )


def test_exactly_100_observations_accepted(client):
    r = upload(client, batch("b-1", 1, observations=obs(100)))
    assert r.status_code == 201
    assert r.json()["observation_count"] == 100


# ---------------- 游标与批次查询 ----------------


def test_cursor_and_batch_queries(client):
    # 未上传过的设备：空链游标，最近事件字段为空
    cur = client.get("/field-devices/dev-empty/cursor").json()
    assert cur["last_sequence"] == 0 and cur["head_digest"] == ""
    assert cur["next_sequence"] == 1 and cur["expected_previous_digest"] == ""
    assert cur["batch_count"] == 0 and cur["updated_at"] is None
    assert cur["last_event_status"] is None
    assert cur["last_event_sequence"] is None
    assert cur["last_event_at"] is None
    assert client.get("/field-devices/dev-empty/batches").json()["count"] == 0

    d = upload(client, batch("b-1", 1)).json()["digest"]
    d = upload(client, batch("b-2", 2, previous_digest=d)).json()["digest"]
    upload(client, batch("b-3", 3, previous_digest=d))

    # 列表：状态字段区分 accepted；next_cursor 为 None 表示无更多；
    # 未重放过的批次 replay_count 为零、last_replayed_at 为空
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()
    assert lst["count"] == 3 and lst["next_cursor"] is None
    assert [b["sequence"] for b in lst["batches"]] == [1, 2, 3]
    assert all(b["status"] == "accepted" for b in lst["batches"])
    assert all(b["replay_count"] == 0 for b in lst["batches"])
    assert all(b["last_replayed_at"] is None for b in lst["batches"])
    assert lst["batches"][0]["batch_id"] == "b-1"

    # 游标：最近事件为最新受理
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_event_status"] == "accepted"
    assert cur["last_event_sequence"] == 3
    assert parse_ts(cur["last_event_at"]) is not None

    # 游标分页：after_sequence 续查
    page1 = client.get(f"/field-devices/{DEVICE}/batches?limit=2").json()
    assert [b["sequence"] for b in page1["batches"]] == [1, 2]
    assert page1["next_cursor"] == 2
    page2 = client.get(
        f"/field-devices/{DEVICE}/batches?after_sequence={page1['next_cursor']}&limit=2"
    ).json()
    assert [b["sequence"] for b in page2["batches"]] == [3]
    assert page2["next_cursor"] is None

    # 详情：观测时间戳与 /judge 四项阈值均已保存，未做判定
    detail = client.get(f"/field-devices/{DEVICE}/batches/2").json()
    assert detail["status"] == "accepted"
    assert detail["replay_count"] == 0 and detail["last_replayed_at"] is None
    assert detail["observations"] == obs(3)
    assert detail["thresholds"] == THRESHOLDS
    assert detail["observation_count"] == 3
    assert "best" not in detail and "candidates" not in detail

    # 不存在 → 404
    assert client.get(f"/field-devices/{DEVICE}/batches/99").status_code == 404
    assert client.get("/field-devices/dev-empty/batches/1").status_code == 404


# ---------------- 并发：不得形成分叉 ----------------


def test_concurrent_same_sequence_no_fork(client):
    """多客户端并发提交同一序号的不同批次：恰好一个被接受，其余 409。"""
    upload(client, batch("b-1", 1))
    d1 = client.get(f"/field-devices/{DEVICE}/cursor").json()["head_digest"]

    results = []
    lock = threading.Lock()

    def submit(i):
        r = upload(client, batch(f"race-{i}", 2, previous_digest=d1))
        with lock:
            results.append(r.status_code)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(201) == 1
    assert results.count(409) == len(threads) - 1

    # 链上该序号只有一条记录，链头唯一
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 2 and cur["batch_count"] == 2
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()["batches"]
    assert [b["sequence"] for b in lst] == [1, 2]


def test_concurrent_identical_replay(client):
    """断网重连后并发重发同一批次：恰好一个 accepted，其余 replayed。"""
    body = batch("b-1", 1)
    results = []
    lock = threading.Lock()

    def submit():
        r = upload(client, body)
        with lock:
            results.append((r.status_code, r.json()["status"]))

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count((201, "accepted")) == 1
    assert results.count((200, "replayed")) == len(threads) - 1
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 1 and cur["batch_count"] == 1


# ---------------- 重放状态持久化 ----------------


def test_replay_state_persisted_sequential(client):
    """顺序重放：重放计数/时刻持久化到列表、详情与游标，新受理后游标复位。"""
    body = batch("b-1", 1)
    r1 = upload(client, body)
    assert r1.status_code == 201

    # 两次相同载荷重发：均返回原结果（status=replayed）
    for _ in range(2):
        r = upload(client, body)
        assert r.status_code == 200 and r.json()["status"] == "replayed"

    # 列表：status 保留首次受理事实，重放计数与最近重放时刻已持久化
    item = client.get(f"/field-devices/{DEVICE}/batches").json()["batches"][0]
    assert item["status"] == "accepted"
    assert item["replay_count"] == 2
    assert parse_ts(item["last_replayed_at"]) >= parse_ts(item["created_at"])

    # 详情：字段含义与列表一致
    detail = client.get(f"/field-devices/{DEVICE}/batches/1").json()
    assert detail["status"] == "accepted"
    assert detail["replay_count"] == item["replay_count"]
    assert detail["last_replayed_at"] == item["last_replayed_at"]

    # 游标：最近事件为 replayed，链头未推进
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_event_status"] == "replayed"
    assert cur["last_event_sequence"] == 1
    assert parse_ts(cur["last_event_at"]) >= parse_ts(cur["updated_at"])
    assert cur["last_sequence"] == 1 and cur["batch_count"] == 1

    # 新批次受理后：游标最近事件改为 accepted
    d1 = r1.json()["digest"]
    r2 = upload(client, batch("b-2", 2, previous_digest=d1))
    assert r2.status_code == 201
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_event_status"] == "accepted"
    assert cur["last_event_sequence"] == 2
    assert cur["last_sequence"] == 2 and cur["batch_count"] == 2

    # 链头前移后重放历史批次：计数累加在旧批次上，最近事件指向它，链头不动
    r = upload(client, body)
    assert r.status_code == 200 and r.json()["status"] == "replayed"
    # 首次受理响应未被改写：重放响应与首次响应仅 status 不同
    assert {k: v for k, v in r.json().items() if k != "status"} == {
        k: v for k, v in r1.json().items() if k != "status"
    }
    detail = client.get(f"/field-devices/{DEVICE}/batches/1").json()
    assert detail["replay_count"] == 3
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_event_status"] == "replayed"
    assert cur["last_event_sequence"] == 1
    assert cur["last_sequence"] == 2 and cur["batch_count"] == 2
    # 未重放过的批次不受影响
    detail2 = client.get(f"/field-devices/{DEVICE}/batches/2").json()
    assert detail2["replay_count"] == 0 and detail2["last_replayed_at"] is None


def test_concurrent_replay_count_not_lost(client):
    """并发重放同一已受理批次：计数不丢失，不新增批次、不推进链头。"""
    body = batch("b-1", 1)
    assert upload(client, body).status_code == 201

    n = 16
    results = []
    lock = threading.Lock()

    def replay():
        r = upload(client, body)
        with lock:
            results.append((r.status_code, r.json()["status"]))

    threads = [threading.Thread(target=replay) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count((200, "replayed")) == n

    # 每次重放都被计数：列表与详情一致
    detail = client.get(f"/field-devices/{DEVICE}/batches/1").json()
    assert detail["status"] == "accepted"
    assert detail["replay_count"] == n
    assert detail["last_replayed_at"] is not None
    item = client.get(f"/field-devices/{DEVICE}/batches").json()["batches"][0]
    assert item["replay_count"] == n

    # 链头未推进、未新增批次；最近事件为重放
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 1 and cur["batch_count"] == 1
    assert cur["last_event_status"] == "replayed"
    assert cur["last_event_sequence"] == 1


def test_concurrent_first_submit_then_replays(client):
    """并发首次提交同一批次：恰一个 accepted，其余 replayed 且计数完整。"""
    body = batch("b-1", 1)
    n = 8
    results = []
    lock = threading.Lock()

    def submit():
        r = upload(client, body)
        with lock:
            results.append((r.status_code, r.json()["status"]))

    threads = [threading.Thread(target=submit) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count((201, "accepted")) == 1
    assert results.count((200, "replayed")) == n - 1

    detail = client.get(f"/field-devices/{DEVICE}/batches/1").json()
    assert detail["replay_count"] == n - 1
    cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cur["last_sequence"] == 1 and cur["batch_count"] == 1
    # 受理必然先于全部重放提交，最近事件为重放
    assert cur["last_event_status"] == "replayed"
    assert cur["last_event_sequence"] == 1


# ---------------- 旧库升级迁移 ----------------

# 上一版本的建表语句：无 replay_count / last_replayed_at / last_event_* 列
OLD_SCHEMA = """
CREATE TABLE library_versions (
    version_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    status       TEXT NOT NULL CHECK (status IN ('draft', 'active', 'historical')),
    base_version INTEGER REFERENCES library_versions(version_id),
    source       TEXT NOT NULL DEFAULT 'api',
    created_at   REAL NOT NULL,
    activated_at REAL
);
CREATE TABLE library_version_aids (
    version_id INTEGER NOT NULL REFERENCES library_versions(version_id),
    aid_id     TEXT NOT NULL,
    period     REAL NOT NULL,
    flashes    TEXT NOT NULL,
    PRIMARY KEY (version_id, aid_id)
);
CREATE TABLE library_state (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    version_id INTEGER NOT NULL REFERENCES library_versions(version_id)
);
CREATE TABLE judgments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      REAL NOT NULL,
    status          TEXT NOT NULL,
    best_aid_id     TEXT,
    matched         INTEGER NOT NULL DEFAULT 0,
    missed          INTEGER NOT NULL DEFAULT 0,
    false_count     INTEGER NOT NULL DEFAULT 0,
    drift           REAL,
    library_version INTEGER,
    request         TEXT NOT NULL,
    response        TEXT NOT NULL
);
CREATE TABLE device_chains (
    device_id     TEXT PRIMARY KEY,
    last_sequence INTEGER NOT NULL,
    head_digest   TEXT NOT NULL,
    batch_count   INTEGER NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE TABLE device_batches (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id         TEXT NOT NULL,
    sequence          INTEGER NOT NULL,
    batch_id          TEXT NOT NULL,
    previous_digest   TEXT NOT NULL,
    digest            TEXT NOT NULL,
    observation_count INTEGER NOT NULL,
    thresholds        TEXT NOT NULL,
    observations      TEXT NOT NULL,
    response          TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'accepted',
    created_at        REAL NOT NULL,
    UNIQUE (device_id, sequence),
    UNIQUE (device_id, batch_id)
);
"""


def test_old_database_upgrade(tmp_path):
    """旧库升级：兼容迁移补列，旧记录重放次数为零，升级后重放正常累计。"""
    db_file = tmp_path / "old.db"
    body = batch("b-1", 1)
    digest = expected_digest(DEVICE, body)
    accepted_at = 1_700_000_000.0
    first_response = {
        "status": "accepted",
        "device_id": DEVICE,
        "batch_id": "b-1",
        "sequence": 1,
        "digest": digest,
        "previous_digest": "",
        "observation_count": 3,
        "received_at": "2023-11-14T22:13:20+00:00",
        "chain": {
            "head_sequence": 1,
            "head_digest": digest,
            "next_sequence": 2,
            "expected_previous_digest": digest,
        },
    }
    with sqlite3.connect(str(db_file)) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.execute(
            "INSERT INTO device_batches "
            "(device_id, sequence, batch_id, previous_digest, digest, "
            "observation_count, thresholds, observations, response, status, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted', ?)",
            (
                DEVICE, 1, "b-1", "", digest, 3,
                json.dumps(THRESHOLDS, ensure_ascii=False),
                json.dumps(obs(3), ensure_ascii=False),
                json.dumps(first_response, ensure_ascii=False),
                accepted_at,
            ),
        )
        conn.execute(
            "INSERT INTO device_chains "
            "(device_id, last_sequence, head_digest, batch_count, updated_at) "
            "VALUES (?, 1, ?, 1, ?)",
            (DEVICE, digest, accepted_at),
        )

    app = create_app(db_path=str(db_file))
    with TestClient(app) as client:
        # 旧记录：重放次数为零、最近重放时刻为空；status 保留首次受理事实
        detail = client.get(f"/field-devices/{DEVICE}/batches/1").json()
        assert detail["status"] == "accepted"
        assert detail["replay_count"] == 0
        assert detail["last_replayed_at"] is None
        item = client.get(f"/field-devices/{DEVICE}/batches").json()["batches"][0]
        assert item["replay_count"] == 0 and item["last_replayed_at"] is None

        # 游标：最近事件回填为链头批次的受理（时刻即链头推进时刻）
        cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
        assert cur["last_event_status"] == "accepted"
        assert cur["last_event_sequence"] == 1
        assert cur["last_event_at"] == cur["updated_at"]

        # 升级后重放正常累计，首次响应未被改写
        r = upload(client, body)
        assert r.status_code == 200
        assert r.json() == {**first_response, "status": "replayed"}
        detail = client.get(f"/field-devices/{DEVICE}/batches/1").json()
        assert detail["replay_count"] == 1
        assert detail["last_replayed_at"] is not None
        cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
        assert cur["last_event_status"] == "replayed"
        assert cur["last_event_sequence"] == 1
        assert cur["last_sequence"] == 1 and cur["batch_count"] == 1

        # 升级后新批次正常受理，游标最近事件复位为 accepted
        r2 = upload(client, batch("b-2", 2, previous_digest=digest))
        assert r2.status_code == 201
        cur = client.get(f"/field-devices/{DEVICE}/cursor").json()
        assert cur["last_event_status"] == "accepted"
        assert cur["last_event_sequence"] == 2
        assert cur["last_sequence"] == 2 and cur["batch_count"] == 2
