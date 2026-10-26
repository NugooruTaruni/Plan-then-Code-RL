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

    # Only unescape if the code does not parse as-is but the unescaped version does
    # (handles JSON-escaped outputs without corrupting valid code that legitimately
    # contains backslash escapes such as "\n" or regex r"\d+").
    try:
        ast.parse(code)
    except SyntaxError:
        try:
            cand = code.encode().decode('unicode_escape')
            ast.parse(cand)
            code = cand
        except Exception:
            pass

    if test_list and len(test_list) > 0:
        first_test = str(test_list[0])
        expected = None
        if 'def check(' in first_test:
            # HumanEval format: the real entry point is the LAST check(<name>) call
            # (the first match is the `def check(candidate)` parameter, not the function).
            cands = re.findall(r"check\(\s*(\w+)\s*\)", first_test)
            if cands:
                expected = cands[-1]
        else:
            m = re.search(r"assert\s+(\w+)\s*\(", first_test)
            if m:
                expected = m.group(1)
        if expected:
            all_defs = re.findall(r"def\s+(\w+)\s*\(", code)
            # Only rename when the expected entry function is NOT already defined,
            # and only rename the LAST top-level def (the entry point) — never helper
            # or nested functions (the previous version renamed ALL defs, collapsing
            # multi-function solutions and breaking them).
            if all_defs and expected not in all_defs:
                tl = list(re.finditer(r"(?m)^def\s+\w+\s*\(", code))
                if tl:
                    last = tl[-1].start()
                    code = code[:last] + re.sub(r"^def\s+\w+\s*\(", f"def {expected}(",
                                                code[last:], count=1)

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
