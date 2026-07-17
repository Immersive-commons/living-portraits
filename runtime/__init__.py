"""
runtime/ -- the deterministic playback layer for living-portraits.

clip_graph.py  : the clip library as a directed graph (poses=nodes, clips=edges),
                 shortest-path traversal, and a JSON manifest KV-index.
clip_player.py : renders a clip / clip-path into one panel surface, crossfading
                 seams, with a synthetic source (no assets needed) and an
                 asset-backed source (opencv decode of mp4 / png-sequence).

Both modules are import-safe with no GPU and no display: pygame and opencv are
import-guarded, and the headless self-tests run against a numpy-backed
DummySurface. player.py swaps its text dev-view for ClipPlayer while keeping the
same stage_state.json consumption shell.
"""
