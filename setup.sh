python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt > /dev/null 2>&1

if [ -f "hjy.jpg" ]; then
    open hjy.jpg
    osascript -e 'tell application "Preview" to activate'
    echo "Ready! "
else
    echo "Ready (it's a pity that your laptop is not Mac,,,)"
fi

echo 'run "source ibivenv/bin/activate" to begin'