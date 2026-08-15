from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Iterator

import zstandard as zstd


@dataclass
class Comment:
    id: str
    parent_id: str | None
    author: str
    body: str
    created_utc: float
    score: int
    is_submitter: bool = False


@dataclass
class Thread:
    post_id: str
    title: str
    selftext: str
    subreddit: str
    created_utc: float
    comments: list[Comment]
    raw_post: dict[str, Any]


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_comment(c: dict[str, Any]) -> Comment | None:
    body = (c.get("body") or "").strip()
    if not body or body in {"[deleted]", "[removed]"}:
        return None
    parent = c.get("parent_id")
    return Comment(
        id=str(c.get("id") or c.get("name") or ""),
        parent_id=str(parent) if parent else None,
        author=str(c.get("author") or "[unknown]"),
        body=body,
        created_utc=_as_float(c.get("created_utc", c.get("created"))),
        score=int(c.get("score") or 0),
        is_submitter=bool(c.get("is_submitter")),
    )


def parse_thread(obj: dict[str, Any]) -> Thread | None:
    post = obj.get("post") or {}
    comments_raw = obj.get("comments") or []
    comments: list[Comment] = []
    for c in comments_raw:
        pc = _parse_comment(c)
        if pc is not None:
            comments.append(pc)
    if not comments:
        return None
    return Thread(
        post_id=str(post.get("id") or post.get("name") or ""),
        title=str(post.get("title") or ""),
        selftext=str(post.get("selftext") or ""),
        subreddit=str(post.get("subreddit") or ""),
        created_utc=_as_float(post.get("created_utc", post.get("created"))),
        comments=comments,
        raw_post=post,
    )


def iter_threads(
    path: str,
    min_comments: int = 50,
    max_comments: int | None = None,
) -> Iterator[Thread]:
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            th = parse_thread(obj)
            if th is None:
                continue
            n = len(th.comments)
            if n < min_comments:
                continue
            if max_comments is not None and n > max_comments:
                continue
            yield th


def load_thread_by_id(path: str, post_id: str, min_comments: int = 0) -> Thread | None:
    for th in iter_threads(path, min_comments=min_comments):
        if th.post_id == post_id:
            return th
    return None


def _fullname_id(cid: str) -> str:
    # Reddit fullnames look like t1_xxx / t3_xxx
    if "_" in cid:
        return cid.split("_", 1)[1]
    return cid


def linearize_comments(thread: Thread) -> list[Comment]:
    """Top-level first; under each top-level, expand children by created_utc (DFS)."""
    by_parent: dict[str, list[Comment]] = {}
    tops: list[Comment] = []
    post_fullname = _fullname_id(thread.post_id)
    post_full = f"t3_{post_fullname}"

    for c in thread.comments:
        pid = c.parent_id or ""
        pid_norm = _fullname_id(pid)
        if pid.startswith("t3_") or pid_norm == post_fullname or pid in {"", post_full}:
            tops.append(c)
        else:
            by_parent.setdefault(pid_norm, []).append(c)
            by_parent.setdefault(pid, []).append(c)

    tops.sort(key=lambda x: (x.created_utc, x.id))
    for k in by_parent:
        by_parent[k].sort(key=lambda x: (x.created_utc, x.id))

    seen: set[str] = set()
    out: list[Comment] = []

    def walk(node: Comment) -> None:
        if node.id in seen:
            return
        seen.add(node.id)
        out.append(node)
        kids = by_parent.get(node.id, []) + by_parent.get(_fullname_id(node.id), [])
        # de-dup kids
        uniq: dict[str, Comment] = {k.id: k for k in kids}
        for kid in sorted(uniq.values(), key=lambda x: (x.created_utc, x.id)):
            walk(kid)

    for t in tops:
        walk(t)

    # orphan comments (broken parent links): append by time
    orphans = [c for c in thread.comments if c.id not in seen]
    orphans.sort(key=lambda x: (x.created_utc, x.id))
    out.extend(orphans)
    return out


def post_text(thread: Thread) -> str:
    parts = [thread.title.strip()]
    if thread.selftext.strip():
        parts.append(thread.selftext.strip())
    return "\n\n".join(parts)


def window_texts(comments: list[Comment], window_size: int) -> list[str]:
    windows: list[str] = []
    for i in range(0, len(comments), window_size):
        chunk = comments[i : i + window_size]
        if not chunk:
            continue
        windows.append("\n\n".join(c.body for c in chunk))
    return windows
