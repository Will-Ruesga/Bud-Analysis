"""JSON + image endpoints for the viewer."""

from flask import Blueprint, abort, jsonify, request, send_file

from services.run_loader import (
    list_runs, load_records, resolve_image, load_changes, save_changes,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/runs")
def runs():
    """Available runs with their trained variants."""
    return jsonify({"runs": list_runs()})


@api_bp.route("/data")
def data():
    """Per-view records for one run + variant head."""
    run = request.args.get("run", "").strip()
    variant = request.args.get("variant", "").strip()
    if not run or not variant:
        abort(400, "run and variant are required")
    try:
        return jsonify(load_records(run, variant))
    except FileNotFoundError as exc:
        abort(404, str(exc))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409


@api_bp.route("/image")
def image():
    """Serve one source image by its record `file_name`, relative to data_dir."""
    run = request.args.get("run", "").strip()
    rel = request.args.get("file", "").strip()
    if not run or not rel:
        abort(400, "run and file are required")
    path = resolve_image(run, rel)
    if path is None:
        abort(404)
    return send_file(path)


@api_bp.route("/changes", methods=["GET"])
def changes_get():
    """Saved label corrections for a run, as `{flower_id: new_class}`."""
    run = request.args.get("run", "").strip()
    if not run:
        abort(400, "run is required")
    return jsonify({"changes": load_changes(run)})


@api_bp.route("/changes", methods=["POST"])
def changes_post():
    """Persist label corrections to output/<run>/ripeness_changes.csv."""
    body = request.get_json(silent=True) or {}
    run = (body.get("run") or "").strip()
    if not run:
        abort(400, "run is required")
    return jsonify(save_changes(run, body.get("changes") or []))
