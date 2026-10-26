import subprocess, tempfile, os
def _run_file(full, timeout=20):
    p=None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(full); p=fh.name
        r=subprocess.run(["python3", p], capture_output=True, text=True, timeout=timeout)
        return r.returncode==0, r.stderr[:400]
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:400]
    finally:
        if p:
            try: os.unlink(p)
            except: pass
def mbpp_run(code, test_list, test_code=""):
    if not code.strip(): return False, "empty"
    if not test_list and not test_code: return False, "no tests"
    full = (code + "\n\n" + test_code) if test_code else (code + "\n\n" + "\n".join(test_list))
    return _run_file(full)
def he_run(code, test):
    if not code.strip(): return False, "empty"
    return _run_file(code + "\n\n" + test)
