"""Tests for the ②/③/④ selection flow: click-to-pick galleries and carry-forward.

The 0.14.0 defects these lock down were all "the app re-selected things behind the
user": captioning re-checked the whole folder, and ④ Export ignored what ③ captioned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app as A


def _rows() -> list[tuple[str, str, str]]:
    return [("/img/a.png", "a.png", "a.png"),
            ("/img/b.png", "b.png", "b.png"),
            ("/img/c.png", "c.png", "c.png")]


# ---------- the click script's contract with the layout ----------

def test_every_picker_id_is_actually_rendered() -> None:
    """The browser script pairs gallery/checkbox/zoom by elem_id.

    A renamed or dropped elem_id would break click-to-select *silently* — the
    checkbox list keeps working, so nothing errors and nothing is logged.
    """
    rendered = {block.elem_id for block in A.demo.blocks.values()
                if getattr(block, "elem_id", None)}
    for group in A.PICKER_IDS:
        for elem_id in group:
            assert elem_id in rendered, f"{elem_id} is in PICKER_IDS but never rendered"


def test_the_picker_script_embeds_those_same_ids() -> None:
    for group in A.PICKER_IDS:
        for elem_id in group:
            assert f'"{elem_id}"' in A._PICKER_SCRIPT


def test_the_picker_script_is_installed_on_the_page() -> None:
    assert A.demo.head and "thumbnail-item" in A.demo.head


# ---------- picker primitives ----------

def test_picker_gallery_marks_selection_first() -> None:
    items = A._picker_gallery(_rows(), ["b.png"])
    assert [label[0] for _, label in items] == [A._PICK_OFF, A._PICK_ON, A._PICK_OFF]


def test_picker_gallery_keeps_paths_and_labels() -> None:
    items = A._picker_gallery(_rows(), [])
    assert items[0] == ("/img/a.png", f"{A._PICK_OFF} a.png")


def test_picker_order_follows_rows_not_click_order() -> None:
    # A CheckboxGroup value must follow its choices; clicking c then a must not
    # produce ["c.png", "a.png"] or Gradio renders the boxes out of order.
    assert A._picker_order(_rows(), ["c.png", "a.png"]) == ["a.png", "c.png"]


def test_picker_order_drops_values_not_in_rows() -> None:
    assert A._picker_order(_rows(), ["a.png", "gone.png"]) == ["a.png"]


def test_picker_mark_returns_plain_gallery_items() -> None:
    assert A._picker_mark(_rows(), ["b.png"]) == A._picker_gallery(_rows(), ["b.png"])


def test_set_zoom_only_flips_allow_preview() -> None:
    # It must not carry a value, or flipping the mode would blank the gallery.
    on, off = A._set_zoom(True), A._set_zoom(False)
    assert on["allow_preview"] is True
    assert off["allow_preview"] is False
    assert "value" not in on


def test_pick_all_and_none() -> None:
    assert A._pick_all(_rows()).value == ["a.png", "b.png", "c.png"]
    assert A._pick_none(_rows()).value == []


def test_pick_captioned_keeps_only_non_empty_sidecars(tmp_path: Path) -> None:
    good, blank, absent = (tmp_path / n for n in ("g.png", "b.png", "n.png"))
    for p in (good, blank, absent):
        p.write_bytes(b"")
    good.with_suffix(".txt").write_text("a caption", encoding="utf-8")
    blank.with_suffix(".txt").write_text("   ", encoding="utf-8")
    rows = [(str(p), str(p), p.name) for p in (good, blank, absent)]
    assert A._pick_captioned(rows).value == [str(good)]


# ---------- carry-forward ③ -> ④ ----------

def test_merge_carry_accumulates_across_folders() -> None:
    # Captioning the prepped sources and then the generated shots is documented
    # workflow; both halves must survive into the export preselection.
    assert A._merge_carry(["/one/a.png"], ["/two/b.png"]) == ["/one/a.png", "/two/b.png"]


def test_merge_carry_dedupes_and_keeps_first_seen_order() -> None:
    assert A._merge_carry(["/a.png", "/b.png"], ["/b.png", "/c.png"]) == \
        ["/a.png", "/b.png", "/c.png"]


def test_merge_carry_handles_empty_state() -> None:
    assert A._merge_carry(None, ["/a.png"]) == ["/a.png"]


def test_export_preselect_uses_the_captioned_subset() -> None:
    images = [Path("/img/a.png"), Path("/img/b.png"), Path("/img/c.png")]
    values, why = A._export_preselect(images, [str(images[0]), str(images[2])])
    assert values == [str(images[0]), str(images[2])]
    assert "2 of 3 preselected" in why


def test_export_preselect_falls_back_to_all_without_a_carry() -> None:
    # ④ must stay standalone on a folder that never went through ③.
    images = [Path("/img/a.png"), Path("/img/b.png")]
    values, why = A._export_preselect(images, [])
    assert values == [str(i) for i in images]
    assert "all checked" in why


def test_export_preselect_falls_back_when_the_carry_is_elsewhere() -> None:
    images = [Path("/img/a.png")]
    values, _ = A._export_preselect(images, ["/somewhere/else.png"])
    assert values == [str(images[0])]


# ---------- folder loading keeps the user's pick ----------

def _folder_with(tmp_path: Path, names: list[str]) -> Path:
    for n in names:
        (tmp_path / n).write_bytes(b"")
    return tmp_path


def test_load_caption_folder_selects_all_by_default(tmp_path: Path) -> None:
    folder = _folder_with(tmp_path, ["a.png", "b.png"])
    _, _, boxes, _ = A.load_caption_folder(str(folder))
    assert boxes.value == ["a.png", "b.png"]


def test_load_caption_folder_preserves_an_existing_pick(tmp_path: Path) -> None:
    # The 0.13.2 bug: do_caption reloaded the folder and re-checked everything,
    # discarding the subset the user had chosen.
    folder = _folder_with(tmp_path, ["a.png", "b.png", "c.png"])
    _, _, boxes, note = A.load_caption_folder(str(folder), selected=["b.png"])
    assert boxes.value == ["b.png"]
    assert "1 selected" in note


def test_load_caption_folder_drops_a_pick_whose_file_vanished(tmp_path: Path) -> None:
    folder = _folder_with(tmp_path, ["a.png"])
    _, _, boxes, _ = A.load_caption_folder(str(folder), selected=["a.png", "deleted.png"])
    assert boxes.value == ["a.png"]


# ---------- ② -> ③ hand-off ----------

class _Shot:
    def __init__(self, sid: str) -> None:
        self.id = sid


class _Result:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.shot = _Shot(path.stem)


def test_send_kept_to_caption_switches_tab_and_preselects(tmp_path: Path) -> None:
    # The reported silence: this handler's note lands on ③, invisible from ②. It must
    # also select ③ and echo a confirmation back onto ②.
    folder = _folder_with(tmp_path, ["a.png", "b.png", "c.png"])
    results = [_Result(folder / n) for n in ("a.png", "b.png", "c.png")]
    # ② keeps shot IDs; ③ picks filenames — the hand-off has to translate.
    tab, out_dir, rows, gallery, boxes, note, sent = A.send_kept_to_caption(
        results, ["a", "c"], str(folder))

    assert tab.selected == "caption"
    assert out_dir == str(folder)
    assert boxes.value == ["a.png", "c.png"]  # rejected shot stays rejected
    assert len(rows) == len(gallery) == 3
    assert "2 kept shot(s)" in note
    assert "2 kept shot(s)" in sent  # the feedback ② was missing entirely


def test_send_kept_to_caption_rejects_an_empty_pick(tmp_path: Path) -> None:
    folder = _folder_with(tmp_path, ["a.png"])
    with pytest.raises(Exception, match="No kept shots selected"):
        A.send_kept_to_caption([_Result(folder / "a.png")], [], str(folder))


def test_gen_gallery_carries_a_pick_across_a_resync(tmp_path: Path) -> None:
    folder = _folder_with(tmp_path, ["a.png", "b.png"])
    results = [_Result(folder / n) for n in ("a.png", "b.png")]
    _, _, keep = A._gen_gallery(results, selected=["b"])
    assert keep.value == ["b"]


def test_gen_gallery_keeps_everything_on_a_fresh_run(tmp_path: Path) -> None:
    folder = _folder_with(tmp_path, ["a.png", "b.png"])
    results = [_Result(folder / n) for n in ("a.png", "b.png")]
    _, _, keep = A._gen_gallery(results)
    assert keep.value == ["a", "b"]


def test_load_export_preview_preselects_the_captioned_subset(tmp_path: Path) -> None:
    folder = _folder_with(tmp_path, ["a.png", "b.png", "c.png"])
    for n in ("a.png", "b.png"):
        (folder / n).with_suffix(".txt").write_text("cap", encoding="utf-8")
    _, _, boxes, note = A.load_export_preview(
        str(folder), carry=[str(folder / "a.png")])
    assert boxes.value == [str(folder / "a.png")]
    assert "1 of 3 preselected" in note
