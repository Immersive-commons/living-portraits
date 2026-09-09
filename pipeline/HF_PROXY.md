# The hf-proxy contract

`pipeline/hf_gen.py` is a **client**. The service it talks to is not in this repo and
not in the private monorepo either — it lives in a separate deployment (`j4me`) on a
host called `node`, which owns the Higgsfield session.

So minting a pose needs four things: **this repo → the proxy → the owner host →
Higgsfield (metered credits)**. Two of them are invisible from here, which is why this
file exists: **the contract below is complete, and you can implement it yourself.**

Nothing is reverse-engineered here. `hf_gen.py` is a full specification of what the
proxy must do, and this is that specification written out.

---

## Read this before you point `LP_HF_PROXY_URL` anywhere

**The session has ONE owner and the refresh tokens are single-use. A second host
running the `hf` backend steals the session and kills the first one.**

That is not a rate limit or a warning — it is how the upstream auth works. It means
this is the one dependency in the system you cannot safely learn by trying, because
the failure mode is taking down production generation for whoever owns it now. Stand
up your own proxy against your own Higgsfield account; do not point at someone else's.

`healthy()` exists for exactly this: a present credentials file with a dead session
renders nothing, so it checks liveness as well as reachability.

---

## Endpoints

Two, both authenticated with `Authorization: Bearer <token>`.

### `POST /run_create`

```jsonc
{
  "model":   "kling3_0",              // vendor model id
  "args":    ["--prompt", "...", "--quality", "high"],
  "exts":    ["mp4", "webm", "mov"],  // acceptable output extensions
  "files":   {"REF.png": "<base64>"}, // inlined, keyed by an opaque name
  "timeout": "15m",                   // minutes, as a string
  "attempts": 1
}
```

Response:

```jsonc
{"ok": true,  "url": "https://.../asset.mp4"}
{"ok": false, "error": "human-readable vendor error"}
```

**Two rules the proxy enforces, and the client depends on both:**

1. **A file flag's value must be a key present in `files`, never a host path.** The
   proxy refuses a path — otherwise a caller could have an arbitrary file on the proxy's
   box uploaded off it. So `args` carries `["--image", "REF.png"]` and `files` carries
   `{"REF.png": "<base64>"}`.
2. **The key must carry a real extension.** The proxy names its temporary file from the
   key's suffix, and the vendor rejects a `.bin` it cannot type (`Cannot detect media
   type`). `REF.png`, not `REF` or `REF.bin`.

### `GET /healthz`

```jsonc
{"ok": true, "session": true}
```

`session: false` means the proxy is up and the **owner's** upstream session is dead —
a different failure from the proxy being unreachable, and it needs re-auth on the owner
host rather than anything here.

## Error taxonomy

The client maps responses to a `kind`, and callers branch on it. `pipeline/autogen.py`
treats `capability`, `refused` and `no_credits` as terminal and re-queues everything
else, so **a proxy that classifies wrongly either burns credits on retries or gives up
on work that would have succeeded.**

| condition | kind | why it matters |
|---|---|---|
| HTTP 401 / 403 | `auth` | Retried. This is what an expired token looks like |
| any other HTTP error, or unreachable | `transient` | Retried |
| `ok:false` containing `medias` / `only contain` / `at most 1 item` | `capability` | **Terminal.** The model cannot take an end frame; retrying learns nothing |
| `ok:false` containing `rate_limit` / `concurrent` | `rate_limit` | Sets a 10-minute cooldown. The plan caps 8 concurrent jobs |
| `ok:false` containing `not_enough_credits` / `insufficient` | `no_credits` | **Backs off until the grant lands.** Left as `transient` it was retried 7,279 times over six days in 2026-08 |
| anything else | `transient` | Retried |

A returned `url` is **not** trusted: `_download()` checks the scheme and fetches
`http`/`https` only, because `urllib` honours `file://` and the URL comes from the
proxy's response rather than from our config. A non-http URL is `refused`, which is
terminal — a proxy that returns one will keep doing so.

## Credentials

Resolved in order, first complete pair wins:

1. `LP_HF_PROXY_URL` + `LP_HF_PROXY_TOKEN` — **both, or it falls through**
2. `~/.config/living-portraits/hf_proxy.json`
3. `data/mind/hf_proxy.json` — gitignored

Each of 2 and 3 is `{"url": "...", "token": "..."}`. See
[`install/ENVIRONMENT.md`](../install/ENVIRONMENT.md).

## What this path costs

`_bakeoff/README.md` has the measurements behind the model choice: eight models handed
three real edges, with what each actually delivered. The short version is
`gpt_image_2` for stills at 4 credits (`--quality high --resolution 1k` — the anchor
every clip of a pose is generated from, so a soft anchor makes every downstream edge
soft) and `kling3_0` for clips at 7.5 with sound off.

Budgets live in `pipeline/autogen.py` (`CLIP_DAILY_CAP` / `CLIP_CHAR_CAP`), and
`credits_used()` reports what the vendor has actually billed this grant period.

## What is deliberately not here

The proxy's **source**. It is a separate deployment with its own auth and its own host,
and publishing it is not this repo's call to make. What is published is everything you
need to write your own: the wire format, both rules it enforces, the error taxonomy the
client depends on, and the health contract.

If you implement it and something in this document turns out to be wrong, that is a bug
in this file — `hf_gen.py` is the authority, and every line here was read off it.
