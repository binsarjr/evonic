from ...parser import parse_nik


def execute(agent: dict, args: dict):
    return parse_nik(args.get("nik")) if "nik" in args else {"error": "NIK_REQUIRED"}
