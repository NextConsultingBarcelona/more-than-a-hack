"""tools/sanity_check.py on a fake match: passes on pipeline output, fails on broken files."""

import json

import sanity_check
from src import io as mio


def _write(tmp_path, labels, name="pred.csv"):
    return mio.write_submission(str(tmp_path / name), labels)


def test_pipeline_output_passes(fake_match, fake_labels, tmp_path):
    sim = fake_match["sim"]
    path = _write(tmp_path, fake_labels[0])
    results = sanity_check.check(path, sim["n_frames"], sim["blocks"], [])
    assert all(ok for _, ok in results), results

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"n_frames": sim["n_frames"], "blocks": sim["blocks"]}))
    assert sanity_check.main([path, "--meta", str(meta)]) == 0


def test_broken_files_fail(fake_match, fake_labels, tmp_path):
    sim = fake_match["sim"]
    labels = fake_labels[0].copy()
    first, last = sim["blocks"][1]

    def failed(path, dead_spans=()):
        return {name for name, ok in sanity_check.check(path, sim["n_frames"], sim["blocks"],
                                                        list(dead_spans)) if not ok}

    bad = labels.copy()
    bad[first] = "X"                                    # abstains inside a half
    assert "ingen X i omgangene" in failed(_write(tmp_path, bad))

    bad = labels.copy()
    bad[first:first + 50] = "H"                         # live before the kick-off
    assert any(n.startswith("D for avspark") for n in failed(_write(tmp_path, bad),
                                                             [(first, first + 49)]))

    assert failed(_write(tmp_path, labels[:-1]))        # one row short

    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(b"\xef\xbb\xbf" + open(_write(tmp_path, labels), "rb").read()
                     .replace(b"\n", b"\r\n"))
    assert {"ingen BOM", "linjeskift er \\n"} <= failed(str(crlf))
