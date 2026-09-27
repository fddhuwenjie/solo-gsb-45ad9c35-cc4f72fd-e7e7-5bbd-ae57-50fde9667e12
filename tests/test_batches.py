"""现场设备批次链接口测试：链式追加、摘要、幂等重放、冲突与并发。"""
import hashlib
import json
import os
import sys
import threading

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
    # 未上传过的设备：空链游标
    cur = client.get("/field-devices/dev-empty/cursor").json()
    assert cur["last_sequence"] == 0 and cur["head_digest"] == ""
    assert cur["next_sequence"] == 1 and cur["expected_previous_digest"] == ""
    assert cur["batch_count"] == 0 and cur["updated_at"] is None
    assert client.get("/field-devices/dev-empty/batches").json()["count"] == 0

    d = upload(client, batch("b-1", 1)).json()["digest"]
    d = upload(client, batch("b-2", 2, previous_digest=d)).json()["digest"]
    upload(client, batch("b-3", 3, previous_digest=d))

    # 列表：状态字段区分 accepted；next_cursor 为 None 表示无更多
    lst = client.get(f"/field-devices/{DEVICE}/batches").json()
    assert lst["count"] == 3 and lst["next_cursor"] is None
    assert [b["sequence"] for b in lst["batches"]] == [1, 2, 3]
    assert all(b["status"] == "accepted" for b in lst["batches"])
    assert lst["batches"][0]["batch_id"] == "b-1"

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
