from ...parser import validate_nik


def execute(agent: dict, args: dict):
    return validate_nik(args.get("nik")) if "nik" in args else {"error": "NIK_REQUIRED"}
