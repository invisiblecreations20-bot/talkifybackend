#!/bin/bash

echo "Starting Quick Test..."

# 1. API Health Check
echo "Checking API Health..."
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health | grep 200 > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ API is UP"
else
    echo "❌ API is DOWN"
    exit 1
fi

# 2. Database Check (Simple Ping using API)
echo "Checking Database Connectivity..."
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/db-check | grep 200 > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Database Connected"
else
    echo "❌ Database Issue"
    exit 1
fi

# 3. Login Test
echo "Testing Login API..."
LOGIN_RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/login \
    -H "Content-Type: application/json" \
    -d '{"email": "test@example.com", "password": "password123"}')

if [[ $LOGIN_RESPONSE == *"token"* ]]; then
    echo "✅ Login Successful"
else
    echo "❌ Login Failed"
    exit 1
fi

# 4. UI Check (Optional: Check if homepage is loading)
echo "Checking UI Home Page..."
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000 | grep 200 > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ UI is UP"
else
    echo "❌ UI is DOWN"
fi

echo "Quick Test Completed!"
