#!/bin/bash

# Get the absolute path of the directory where THIS script is located
INSTALL_DIR=$(cd "$(dirname "$0")" && pwd)

APP_NAME="Honeychrome"
EXE_PATH="$INSTALL_DIR/honeychrome/honeychrome"
# Point to the icon inside the portable package's asset folder
ICON_PATH="$INSTALL_DIR/honeychrome/_internal/honeychrome/view_components/assets/cytkit_web_logo.png"
INTERNAL_NAME="honeychrome"

DESKTOP_FILE_PATH="$HOME/.local/share/applications/$INTERNAL_NAME.desktop"

echo "Installing $APP_NAME shortcut from $INSTALL_DIR..."

cat <<EOF > "$DESKTOP_FILE_PATH"
[Desktop Entry]
Version=1.0.0
Type=Application
Name=$APP_NAME
Comment=Honeychrome Open Source Cytometry Acquisition and Analysis
Exec=$EXE_PATH
Icon=$ICON_PATH
Terminal=false
StartupWMClass=$INTERNAL_NAME
Categories=Utility;Science;
EOF

chmod +x "$DESKTOP_FILE_PATH"
chmod +x "$EXE_PATH"

update-desktop-database ~/.local/share/applications/
echo "Success! You can now find $APP_NAME in your application menu."
