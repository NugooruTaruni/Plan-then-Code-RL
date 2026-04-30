import re, ast


def clean_code(code, test_list=None):
    if test_list is not None:
        try:
            if hasattr(test_list, "tolist"):
                test_list = test_list.tolist()
            elif not isinstance(test_list, list):
                test_list = list(test_list)
        except:
            test_list = []

    code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.MULTILINE)
    code = re.sub(r"\s*```$", "", code, flags=re.MULTILINE)
    code = code.encode().decode('unicode_escape')

    if test_list and len(test_list) > 0:
        first_test = str(test_list[0])
        m = re.search(r"assert\s+(\w+)\(", first_test)
        if m:
            expected = m.group(1)
            code = re.sub(
                r"def\s+(\w+)\s*\(",
                lambda m: f"def {expected}(" if m.group(1) != expected else m.group(0),
                code
            )

    code = code.replace("\t", "    ")

    try:
        ast.parse(code)
    except SyntaxError:
        lines = code.split('\n')
        fixed = []
        for line in lines:
            stripped = line.lstrip()
            if (stripped.startswith(('def ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:'))
                    and not stripped.rstrip().endswith(':')):
                line = line.rstrip() + ':'
            fixed.append(line)
        code = '\n'.join(fixed)
    return code.strip()
