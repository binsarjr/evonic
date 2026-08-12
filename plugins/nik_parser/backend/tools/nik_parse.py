from ._bridge import module

def execute(agent: dict, args: dict):
    return module("parser").parse_nik(args.get("nik")) if "nik" in args else {"error": "NIK_REQUIRED"}
