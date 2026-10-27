#!/bin/sh

# --- CONFIGURATION ---
APP_NAME="honeychrome"
VERSION="1.0.0"
DMG_NAME="${APP_NAME}-v${VERSION}-macos.dmg"
VOL_NAME="${APP_NAME} Installer"

# Paths
SOURCE_APP="dist/honeychrome.app"
APP_BUNDLE="${APP_NAME}.app"
BACKGROUND="other/honeychrome_installer_background.png"
ICON_FILE="src/honeychrome/view_components/assets/cytkit_web_logo.png"

# Remove existing DMG if it exists
if [ -f "$DMG_NAME" ]; then
  rm "$DMG_NAME"
fi

# Step 1: Sign the .app bundle BEFORE creating the DMG
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: OLIVER TEAL BURTON (S673JAMG2N)" \
  --options runtime \
  --entitlements entitlements.plist \
  "$SOURCE_APP"

# Step 2: Create the DMG
create-dmg \
  --volname "$VOL_NAME" \
  --volicon "$ICON_FILE" \
  --background "$BACKGROUND" \
  --window-pos 200 120 \
  --window-size 770 450 \
  --icon-size 100 \
  --icon "$APP_BUNDLE" 140 250 \
  --hide-extension "$APP_BUNDLE" \
  --app-drop-link 620 250 \
  --skip-jenkins \
  --eula "LICENSE.txt" \
  "$DMG_NAME" \
  "$SOURCE_APP"

# Step 3: Sign the DMG
codesign --sign "Developer ID Application: OLIVER TEAL BURTON (S673JAMG2N)" "$DMG_NAME"

# Step 4: Notarize the DMG
xcrun notarytool submit "$DMG_NAME" \
  --apple-id "$APPLE_ID" \
  --password "$APPLE_APP_PASSWORD" \
  --team-id "$APPLE_TEAM_ID" \
  --wait

# Step 5: Staple the notarization ticket to the DMG with retries
MAX_RETRIES=5
COUNT=0
SUCCESS=false

while [ $COUNT -lt $MAX_RETRIES ]; do
  echo "Attempting to staple... (Attempt $((COUNT+1))/$MAX_RETRIES)"
  if xcrun stapler staple "$DMG_NAME"; then
    echo "Staple successful!"
    SUCCESS=true
    break
  else
    COUNT=$((COUNT+1))
    echo "Staple failed. Retrying in 30 seconds..."
    sleep 30
  fi
done

if [ "$SUCCESS" = false ]; then
  echo "Stapling failed after $MAX_RETRIES attempts."
  exit 1
fi

xcrun stapler validate "$DMG_NAME"