# diagnose_choice_env.py
import os, sys, traceback, platform, ctypes, pathlib
from datetime import datetime

HERE = pathlib.Path(__file__).parent.resolve()
USER_HOME = pathlib.Path.home()
DESKTOP = USER_HOME / "Desktop"

def p(msg): print(msg, flush=True)

def add_dll_search_path(d):
    try:
        d = str(pathlib.Path(d).resolve())
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
            p(f"[OK] add_dll_directory: {d}")
        else:
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH","")
            p(f"[OK] prepend PATH: {d}")
    except Exception as e:
        p(f"[WARN] add_dll_directory failed for {d}: {e}")

def check_files():
    p("=== Basic Info ===")
    p(f"Python: {sys.version.split()[0]}  ({platform.architecture()[0]})")
    p(f"cwd   : {os.getcwd()}")
    p(f"HERE  : {HERE}")
    p(f"HOME  : {USER_HOME}")
    p(f"DESK  : {DESKTOP}")
    p("")

    # 期待的文件位置（和本脚本同目录）
    files = {
        "EmQuantAPI.py": HERE / "EmQuantAPI.py",
        "EmQuantAPI_x64.dll": HERE / "EmQuantAPI_x64.dll",
        "EmQuantAPI.dll": HERE / "EmQuantAPI.dll",
        "userInfo@HOME": USER_HOME / "userInfo",
        "userInfo@DESKTOP": DESKTOP / "userInfo",
        "ServerList.json.e@HERE": HERE / "ServerList.json.e",
    }
    for k, v in files.items():
        p(f"[{'OK' if v.exists() else 'NO'}] {k}: {v}")
    p("")

    # DLL 搜索路径（把脚本目录加入）
    add_dll_search_path(HERE)

def try_login(param_str):
    from EmQuantAPI import c
    p(f"\n--- c.start({param_str!r}) ---")
    try:
        code = c.start(param_str) if param_str else c.start()
        p(f"c.start return code = {code}")
        if code == 0:
            p("[OK] 登录成功")
            c.stop()
            return True
        else:
            p("[ERR] 登录失败，返回码非 0")
            try:
                p("c.geterror(): " + str(c.geterror()))
            except Exception:
                pass
            c.stop()
            return False
    except Exception as e:
        p(f"[EXCEPTION] {e}")
        traceback.print_exc()
        return False

def main():
    check_files()

    # 组合一：默认（要求 userInfo 在用户根目录）
    if try_login(""): return

    # 组合二：显式 Desktop 路径
    ui_desktop = str((DESKTOP / "userInfo").resolve())
    if try_login(f"UserInfo={ui_desktop},ForceLogin=1"): return

    # 组合三：显式 Home 路径 + LogFile
    ui_home = str((USER_HOME / "userInfo").resolve())
    logf = str((HERE / "emquant_login.log").resolve())
    if try_login(f"UserInfo={ui_home},LogFile={logf},ForceLogin=1"): return

    # 组合四：如果你把 ServerList.json.e 放在脚本目录，也显式指定
    sv = HERE / "ServerList.json.e"
    if sv.exists():
        if try_login(f"UserInfo={ui_home},ServerList={sv.resolve()},LogFile={logf},ForceLogin=1"): return

    p("\n=== 建议排查 ===")
    p("1) 确认 Python 位数与 DLL 匹配：64 位 Python 用 EmQuantAPI_x64.dll；32 位 Python 用 EmQuantAPI.dll")
    p("2) 确认 EmQuantAPI.py 与 DLL 和本脚本在同一目录，或位于 site-packages/EmQuantAPI/ 下")
    p("3) 确认 userInfo 在：C:\\Users\\<用户名>\\userInfo ；或改用上面显式 UserInfo=... 的方式")
    p("4) 如仍报 WinError 87，通常是 DLL 依赖缺失：安装“Microsoft Visual C++ 2015-2022 x64 运行库”")
    p("5) 若公司安全软件拦截，尝试以管理员权限运行/把目录加入白名单")
    p("6) 若路径含特殊字符/中文，尽量放到简单路径（例如 C:\\EmQuant\\），并使用双反斜杠或原始字符串")

if __name__ == "__main__":
    main()
