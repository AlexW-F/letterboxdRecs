"""Shared multi-device group state.

A ``Group`` is a server-side container with:
- A short URL-safe ID (e.g. ``gx7m4q2k``) — the basis of the shareable join link.
- A list of members (each is `{name, hash, ...}` referring to an existing
  upload-letterboxd cache entry).
- A vote ledger: ``{movie_id: {member_name: 'up'|'veto'}}``.

Stored under the existing diskcache with key ``group::{group_id}`` so we
inherit the same on-disk persistence model as uploads — no new datastore
needed for the MVP. Groups are append-only (members can join + vote; no
remove flow yet).
"""

from __future__ import annotations

import logging
import secrets
import time

from fastapi import APIRouter, HTTPException, Request

from ..schemas import (
    GroupCreateRequest,
    GroupJoinRequest,
    GroupMemberOut,
    GroupStateOut,
    GroupVoteRequest,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/group", tags=["groups"])


# 8 chars of base32 ≈ 40 bits ≈ 1.1 trillion groups before birthday-collision risk
_ID_ALPHABET = "abcdefghjkmnpqrstvwxyz23456789"  # no 0/o/1/i/l/u confusion


def _new_group_id() -> str:
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(8))


def _state(request: Request):
    s = getattr(request.app.state, "models", None)
    if s is None:
        raise HTTPException(status_code=503, detail="API still warming up")
    return s


def _key(group_id: str) -> str:
    return f"group::{group_id}"


def _load_group(state, group_id: str) -> dict:
    key = _key(group_id)
    if key not in state.cache:
        raise HTTPException(status_code=404, detail=f"unknown group {group_id!r}")
    return state.cache[key]


def _save_group(state, group_id: str, data: dict) -> None:
    state.cache[_key(group_id)] = data


def _to_out(group: dict) -> GroupStateOut:
    return GroupStateOut(
        group_id=group["group_id"],
        created_at=group["created_at"],
        members=[GroupMemberOut(**m) for m in group.get("members", [])],
        votes=group.get("votes", {}),
    )


# ---------------------------------------------------------------------------
# Create / fetch
# ---------------------------------------------------------------------------


# Demo seed: three cached uploads so first-time visitors can see the
# experience without having to wait through a real upload. Hashes refer
# to existing entries in the diskcache. If they go missing (cache wiped
# / fresh deployment), the /demo endpoint 503s with a useful message.
DEMO_MEMBERS: list[tuple[str, str]] = [
    ("alex (cinephile)",   "278f2d37600571dc3af2ea7c64ed5e15ad0424b48d33918a6a92eb834403d911"),
    ("sara (cinephile)",   "51fe5049194116fbae62aa301454acef0763d8564695e0b324c62ecd798ed47d"),
    ("mike (drama)",       "a1c51fa4adde1e0269d1484e0b786cb1e9fdbf591b794a76f79f0826d5f629d1"),
]


@router.post("/demo", response_model=GroupStateOut)
def create_demo_group(request: Request) -> GroupStateOut:
    """Spin up a fresh shared group seeded with the three cached demo
    members so first-time visitors can see the group experience without
    uploading anything. Each call mints a new group_id so demo users
    don't share vote state."""
    state = _state(request)
    missing = [name for name, h in DEMO_MEMBERS if h not in state.cache]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(f"demo data not in cache (missing: {', '.join(missing)}). "
                    "The operator needs to seed demo uploads first."),
        )
    for _ in range(5):
        gid = _new_group_id()
        if _key(gid) not in state.cache:
            break
    else:
        raise HTTPException(status_code=500, detail="group ID generation failed")
    now = time.time()
    members = []
    for name, h in DEMO_MEMBERS:
        upload = state.cache[h]
        members.append({
            "name": name,
            "hash": h,
            "n_ratings_mapped": len(upload.get("ratings") or {}),
            "n_watchlist": len(upload.get("watchlist_movie_ids") or []),
            "source": upload.get("source", "csv"),
            "letterboxd_username": upload.get("letterboxd_username"),
            "joined_at": now,
        })
    data = {
        "group_id": gid,
        "created_at": now,
        "name": "demo",
        "members": members,
        "votes": {},
    }
    _save_group(state, gid, data)
    return _to_out(data)


@router.post("", response_model=GroupStateOut)
def create_group(req: GroupCreateRequest, request: Request) -> GroupStateOut:
    """Create a new empty group. Returns the ID for the shareable URL."""
    state = _state(request)
    # 5 tries is plenty; collisions are vanishingly rare.
    for _ in range(5):
        gid = _new_group_id()
        if _key(gid) not in state.cache:
            break
    else:
        raise HTTPException(status_code=500, detail="group ID generation failed")
    data = {
        "group_id": gid,
        "created_at": time.time(),
        "name": (req.name or "").strip(),
        "members": [],
        "votes": {},
    }
    _save_group(state, gid, data)
    return _to_out(data)


@router.get("/{group_id}", response_model=GroupStateOut)
def get_group(group_id: str, request: Request) -> GroupStateOut:
    state = _state(request)
    return _to_out(_load_group(state, group_id))


# ---------------------------------------------------------------------------
# Join / leave
# ---------------------------------------------------------------------------


@router.post("/{group_id}/join", response_model=GroupStateOut)
def join_group(group_id: str, req: GroupJoinRequest, request: Request) -> GroupStateOut:
    """Add a member (existing upload hash + display name) to a group.

    Idempotent on (name, hash): joining twice with the same pair is a no-op
    but joining with a new hash under the same name *replaces* the previous
    upload (useful when re-uploading a refreshed CSV)."""
    state = _state(request)
    # transact() makes the read-modify-write atomic so concurrent joins from
    # different phones don't clobber each other (lost-update race).
    with state.cache.transact():
        group = _load_group(state, group_id)
        # Pull metadata from the underlying upload entry so the group page can
        # render member chips with rating counts etc. without N extra requests.
        if req.hash not in state.cache:
            raise HTTPException(status_code=400,
                                detail=f"unknown upload hash {req.hash!r}; "
                                       "upload via /upload-letterboxd[-username] first")
        upload = state.cache[req.hash]
        new_member = {
            "name": req.name.strip(),
            "hash": req.hash,
            "n_ratings_mapped": len(upload.get("ratings") or {}),
            "n_watchlist": len(upload.get("watchlist_movie_ids") or []),
            "source": upload.get("source", "csv"),
            "letterboxd_username": upload.get("letterboxd_username"),
            "joined_at": time.time(),
        }
        members = list(group.get("members", []))
        # Replace any existing entry under the same display name (idempotent re-join)
        members = [m for m in members if m["name"] != new_member["name"]]
        members.append(new_member)
        group["members"] = members
        _save_group(state, group_id, group)
    return _to_out(group)


@router.post("/{group_id}/leave/{member_name}", response_model=GroupStateOut)
def leave_group(group_id: str, member_name: str, request: Request) -> GroupStateOut:
    """Remove a member from the group + clear any of their votes."""
    state = _state(request)
    with state.cache.transact():
        group = _load_group(state, group_id)
        group["members"] = [m for m in group.get("members", []) if m["name"] != member_name]
        # Also clear any votes the leaving member cast
        votes = group.get("votes") or {}
        for mid in list(votes):
            if member_name in votes[mid]:
                del votes[mid][member_name]
                if not votes[mid]:
                    del votes[mid]
        group["votes"] = votes
        _save_group(state, group_id, group)
    return _to_out(group)


# ---------------------------------------------------------------------------
# Shortlist votes (Phase B.3)
# ---------------------------------------------------------------------------


@router.post("/{group_id}/vote", response_model=GroupStateOut)
def cast_vote(group_id: str, req: GroupVoteRequest, request: Request) -> GroupStateOut:
    """Cast a thumbs-up or veto on a film. ``vote="clear"`` removes a prior vote.

    The film doesn't need to be in any pre-defined shortlist — the UI just
    surfaces votes on whichever films it chooses to show. This keeps the
    server stateless about which films are "candidates"."""
    state = _state(request)
    with state.cache.transact():
        group = _load_group(state, group_id)
        votes = group.get("votes") or {}
        mid = str(req.movie_id)
        member_votes = votes.get(mid, {})
        if req.vote == "clear":
            member_votes.pop(req.member_name, None)
        else:
            member_votes[req.member_name] = req.vote
        if member_votes:
            votes[mid] = member_votes
        else:
            votes.pop(mid, None)
        group["votes"] = votes
        _save_group(state, group_id, group)
    return _to_out(group)
