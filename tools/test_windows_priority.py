"""Check the engine's opt-in Windows priority code without CUDA or a model.

Run from an x64 Visual Studio developer prompt: python tools/test_windows_priority.py
"""
import pathlib
import subprocess
import tempfile


def main():
    source = (pathlib.Path(__file__).resolve().parents[1] / "src/program/generate.cpp").read_text()
    entry = source.split("int main(int argc, char** argv) {", 1)[1]
    code, dispatch = entry.split("#endif", 1)
    assert code.startswith("\n#ifdef _WIN32")
    assert dispatch.lstrip().startswith('if (argc > 1 && std::string(argv[1]) == "--expert-gpu-worker")')
    harness = r'''
#include <windows.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
int calls = 0;
bool refuse = false;
BOOL set_priority(HANDLE process, DWORD priority) {
    ++calls;
    if (refuse) { SetLastError(ERROR_ACCESS_DENIED); return FALSE; }
    return SetPriorityClass(process, priority);
}
#define SetPriorityClass set_priority
int apply_priority() {
'''
    harness += code + "\n#endif\nreturn 0;\n}\n#undef SetPriorityClass\n"
    harness += r'''
int main() {
    SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
    _putenv_s("STRATA_ABOVE_NORMAL", "");
    assert(apply_priority() == 0 && calls == 0);
    _putenv_s("STRATA_ABOVE_NORMAL", "0");
    assert(apply_priority() == 0 && calls == 0);
    assert(GetPriorityClass(GetCurrentProcess()) == NORMAL_PRIORITY_CLASS);
    _putenv_s("STRATA_ABOVE_NORMAL", "true");
    assert(apply_priority() == 2 && calls == 0);
    _putenv_s("STRATA_ABOVE_NORMAL", "1");
    assert(apply_priority() == 0 && calls == 1);
    assert(GetPriorityClass(GetCurrentProcess()) == ABOVE_NORMAL_PRIORITY_CLASS);
    refuse = true;
    assert(apply_priority() == 1 && calls == 2);
    puts("Windows priority: unset, disabled, invalid, AboveNormal, API failure OK");
}
'''
    with tempfile.TemporaryDirectory(prefix="strata-priority-") as directory:
        path = pathlib.Path(directory)
        (path / "test.cpp").write_text(harness)
        subprocess.run(["cl", "/nologo", "/EHsc", "/std:c++17", "test.cpp", "/Fe:test.exe"],
                       cwd=path, check=True, capture_output=True)
        subprocess.run([str(path / "test.exe")], check=True)


if __name__ == "__main__":
    main()
