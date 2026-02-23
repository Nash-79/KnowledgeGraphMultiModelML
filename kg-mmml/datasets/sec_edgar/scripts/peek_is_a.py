import json, pathlib
j=json.loads(pathlib.Path("reports/tables/srs_kge_debug.json").read_text())
print("is-a edges:", j["counts"]["edges_by_type"].get("is-a"))
print("HP:", j["scores"]["HP"], "AtP:", j["scores"]["AtP"], "AP:", j["scores"]["AP"])