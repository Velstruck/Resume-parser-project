@echo off
setlocal enabledelayedexpansion

for /f "tokens=1,2 delims=," %%a in (dependencies.csv) do (
    set package=%%a
    set installer=pip
    if "%%b" neq "" (
        set installer=%%b
    )
    echo Installing !package! using !installer!...
    !installer! install !package!
)

endlocal