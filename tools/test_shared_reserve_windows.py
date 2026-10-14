"""Test the actual Windows reserve_shared helper without CUDA or a large allocation.

Run from an x64 Visual Studio developer prompt: python tools/test_shared_reserve_windows.py
"""
import pathlib
import subprocess
import tempfile


def main():
    source = (pathlib.Path(__file__).resolve().parents[1] / "src/core/pinned.cu").read_text()
    helper = source.split("void* reserve_shared(", 1)[1].split("\n#endif", 1)[0]
    harness = r'''
#include <windows.h>
#include <cassert>
#include <cstdint>
#include <string>
#include <cstdio>
DWORD failure = 0;
bool fail_commit = false;
int creates = 0;
HANDLE mapping(HANDLE file, void* security, DWORD flags, DWORD hi, DWORD lo, const wchar_t* name) {
    ++creates;
    if (failure && (flags & SEC_COMMIT)) { SetLastError(failure); return nullptr; }
    return CreateFileMappingW(file, (SECURITY_ATTRIBUTES*)security, flags, hi, lo, name);
}
void* commit(void* p, SIZE_T n, DWORD flags, DWORD protect) {
    if (fail_commit) { SetLastError(ERROR_COMMITMENT_LIMIT); return nullptr; }
    return VirtualAlloc(p, n, flags, protect);
}
#define CreateFileMappingW mapping
#define VirtualAlloc commit
'''
    harness += "void* reserve_shared(" + helper
    harness += r'''
#undef CreateFileMappingW
#undef VirtualAlloc
int main() {
    const std::string name = "Local\\strata-reserve-test-" + std::to_string(GetCurrentProcessId());
    for (DWORD error : {DWORD(0), DWORD(ERROR_NO_SYSTEM_RESOURCES), DWORD(ERROR_COMMITMENT_LIMIT)}) {
        failure = error; creates = 0;
        void* section = nullptr;
        std::string note;
        auto p = (uint8_t*)reserve_shared(65536, name, section, note);
        assert(p && section && creates == (error ? 2 : 1));
        assert((note.find("committed after reservation") != std::string::npos) == (error != 0));
        p[0] = 42; p[65535] = 19;
        HANDLE other = OpenFileMappingW(FILE_MAP_READ, FALSE, std::wstring(name.begin(), name.end()).c_str());
        assert(other);
        auto view = (const uint8_t*)MapViewOfFile(other, FILE_MAP_READ, 0, 0, 65536);
        assert(view && view[0] == 42 && view[65535] == 19);
        UnmapViewOfFile(view); CloseHandle(other);
        UnmapViewOfFile(p); CloseHandle(section);
    }
    failure = ERROR_NO_SYSTEM_RESOURCES; fail_commit = true;
    void* section = nullptr;
    std::string note;
    assert(!reserve_shared(65536, name, section, note) && !section);
    assert(note.find("commit refused (error 1455)") != std::string::npos);
    assert(!OpenFileMappingW(FILE_MAP_READ, FALSE, std::wstring(name.begin(), name.end()).c_str()));
    failure = ERROR_ACCESS_DENIED; creates = 0; fail_commit = false;
    assert(!reserve_shared(65536, name, section, note) && creates == 1);
    puts("shared reserve: normal, 1450, 1455, shared bytes, cleanup, access denied OK");
}
'''
    with tempfile.TemporaryDirectory(prefix="strata-reserve-") as directory:
        path = pathlib.Path(directory)
        (path / "test.cpp").write_text(harness)
        subprocess.run(["cl", "/nologo", "/EHsc", "/std:c++17", "test.cpp", "/Fe:test.exe"],
                       cwd=path, check=True, capture_output=True)
        subprocess.run([str(path / "test.exe")], check=True)


if __name__ == "__main__":
    main()
