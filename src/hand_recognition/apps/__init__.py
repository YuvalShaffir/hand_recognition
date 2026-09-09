"""Front-ends. Deliberately empty: re-exporting `DesktopApp` here would run
`desktop.py` - and so `import pyautogui` - for anyone importing any other
module in this package, including `overlay.py`, which the web app needs and
which must stay importable on a headless deploy. Import the front-end you
want from its own module.
"""
