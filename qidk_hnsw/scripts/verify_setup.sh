#!/bin/bash
# Setup Verification Script for QNN RAG Demo
# This script checks if all prerequisites are properly installed and configured

set +e  # Don't exit on error, we want to check everything

echo "========================================"
echo "  QNN RAG Demo - Setup Verification"
echo "========================================"
echo ""

ERRORS=0
WARNINGS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

echo "1. Checking Environment Variables"
echo "-----------------------------------"

# Check ANDROID_NDK_ROOT
if [ -z "$ANDROID_NDK_ROOT" ]; then
    check_fail "ANDROID_NDK_ROOT is not set"
    echo "   Set it with: export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653"
else
    check_pass "ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
    
    # Verify path exists
    if [ -d "$ANDROID_NDK_ROOT" ]; then
        check_pass "ANDROID_NDK_ROOT directory exists"
        
        # Check for ndk-build
        if [ -f "$ANDROID_NDK_ROOT/ndk-build" ]; then
            check_pass "ndk-build found"
        else
            check_warn "ndk-build not found at expected location"
        fi
        
        # Check version (r25c = 25.2.9519653)
        if [[ "$ANDROID_NDK_ROOT" == *"25.2.9519653"* ]]; then
            check_pass "NDK version appears to be r25c (25.2.9519653)"
        else
            check_warn "NDK version may not be r25c (required: 25.2.9519653)"
            echo "   Current: $ANDROID_NDK_ROOT"
        fi
    else
        check_fail "ANDROID_NDK_ROOT directory does not exist"
    fi
fi

# Check QNN_SDK_ROOT
if [ -z "$QNN_SDK_ROOT" ]; then
    check_fail "QNN_SDK_ROOT is not set"
    echo "   Set it with: export QNN_SDK_ROOT=~/qualcomm/qnn"
else
    check_pass "QNN_SDK_ROOT: $QNN_SDK_ROOT"
    
    # Verify path exists
    if [ -d "$QNN_SDK_ROOT" ]; then
        check_pass "QNN_SDK_ROOT directory exists"
        
        # Check for key directories
        if [ -d "$QNN_SDK_ROOT/bin" ]; then
            check_pass "QNN SDK bin/ directory found"
        else
            check_fail "QNN SDK bin/ directory not found"
        fi
        
        if [ -d "$QNN_SDK_ROOT/lib" ]; then
            check_pass "QNN SDK lib/ directory found"
        else
            check_fail "QNN SDK lib/ directory not found"
        fi
        
        if [ -d "$QNN_SDK_ROOT/include" ]; then
            check_pass "QNN SDK include/ directory found"
        else
            check_fail "QNN SDK include/ directory not found"
        fi
        
        # Check for QNN tools
        if [ -f "$QNN_SDK_ROOT/bin/qnn-onnx-converter" ]; then
            check_pass "qnn-onnx-converter found"
        else
            check_fail "qnn-onnx-converter not found"
        fi
        
        if [ -f "$QNN_SDK_ROOT/bin/qnn-model-lib-generator" ]; then
            check_pass "qnn-model-lib-generator found"
        else
            check_fail "qnn-model-lib-generator not found"
        fi
    else
        check_fail "QNN_SDK_ROOT directory does not exist"
    fi
fi

echo ""
echo "2. Checking System Tools"
echo "-----------------------------------"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    check_pass "Python3 installed: $PYTHON_VERSION"
    
    # Check Python version (should be 3.8-3.10)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ] && [ "$PYTHON_MINOR" -le 10 ]; then
        check_pass "Python version is compatible (3.8-3.10)"
    else
        check_warn "Python version may not be optimal (recommended: 3.8-3.10)"
    fi
else
    check_fail "Python3 not found"
    echo "   Install with: sudo apt install python3"
fi

# Check pip
if command -v pip3 &> /dev/null; then
    check_pass "pip3 installed"
else
    check_fail "pip3 not found"
    echo "   Install with: sudo apt install python3-pip"
fi

# Check ADB
if command -v adb &> /dev/null; then
    ADB_VERSION=$(adb --version 2>&1 | head -1)
    check_pass "ADB installed: $ADB_VERSION"
    
    # Check device connection
    DEVICE_COUNT=$(adb devices | grep -c "device$")
    if [ "$DEVICE_COUNT" -gt 0 ]; then
        check_pass "Android device connected ($DEVICE_COUNT device(s))"
        
        # List devices
        echo "   Connected devices:"
        adb devices | grep "device$" | while read -r line; do
            echo "     - $line"
        done
    else
        check_warn "No Android device connected"
        echo "   Connect device and enable USB debugging"
    fi
else
    check_fail "ADB not found"
    echo "   Install with: sudo apt install adb"
fi

# Check git
if command -v git &> /dev/null; then
    check_pass "Git installed"
else
    check_warn "Git not found (optional but recommended)"
fi

# Check curl
if command -v curl &> /dev/null; then
    check_pass "curl installed"
else
    check_warn "curl not found (needed for dataset download)"
    echo "   Install with: sudo apt install curl"
fi

echo ""
echo "3. Checking Project Structure"
echo "-----------------------------------"

# Check for key directories and files
REQUIRED_DIRS=("android" "data" "models" "prepare" "qnn" "scripts")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "Directory exists: $dir/"
    else
        check_fail "Directory missing: $dir/"
    fi
done

REQUIRED_FILES=("prepare/create_model.py" "qnn/convert_to_qnn.sh" "scripts/build.sh" "scripts/deploy.sh" "requirements.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "File exists: $file"
    else
        check_fail "File missing: $file"
    fi
done

# Check for datasets
echo ""
echo "4. Checking Datasets"
echo "-----------------------------------"

if [ -d "data/siftsmall" ]; then
    check_pass "siftsmall dataset directory exists"
    
    if [ -f "data/siftsmall/siftsmall_base.fvecs" ]; then
        SIZE=$(du -h data/siftsmall/siftsmall_base.fvecs | cut -f1)
        check_pass "siftsmall_base.fvecs found ($SIZE)"
    else
        check_warn "siftsmall_base.fvecs not found"
        echo "   Download with: curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
    fi
    
    if [ -f "data/siftsmall/siftsmall_query.fvecs" ]; then
        check_pass "siftsmall_query.fvecs found"
    else
        check_warn "siftsmall_query.fvecs not found"
    fi
else
    check_warn "siftsmall dataset directory not found"
    echo "   Create with: mkdir -p data/siftsmall"
fi

if [ -d "data/sift" ]; then
    check_pass "sift dataset directory exists"
    
    if [ -f "data/sift/sift_base.fvecs" ]; then
        SIZE=$(du -h data/sift/sift_base.fvecs | cut -f1)
        check_pass "sift_base.fvecs found ($SIZE)"
    else
        check_warn "sift_base.fvecs not found"
        echo "   Download with: curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fi
    
    if [ -f "data/sift/sift_query.fvecs" ]; then
        check_pass "sift_query.fvecs found"
    else
        check_warn "sift_query.fvecs not found"
    fi
else
    check_warn "sift dataset directory not found"
    echo "   Create with: mkdir -p data/sift"
fi

# Check Python dependencies
echo ""
echo "5. Checking Python Dependencies"
echo "-----------------------------------"

if [ -f "requirements.txt" ]; then
    check_pass "requirements.txt found"
    
    if [ -d ".venv" ]; then
        check_pass "Virtual environment exists"
        
        # Activate and check packages
        source .venv/bin/activate 2>/dev/null
        
        PACKAGES=("onnx" "numpy" "protobuf")
        for pkg in "${PACKAGES[@]}"; do
            if python3 -c "import $pkg" 2>/dev/null; then
                VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
                check_pass "$pkg installed: $VERSION"
            else
                check_warn "$pkg not installed"
            fi
        done
        
        deactivate 2>/dev/null
    else
        check_warn "Virtual environment not created yet"
        echo "   Will be created automatically during build"
    fi
else
    check_fail "requirements.txt not found"
fi

# Summary
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You're ready to build and run the demo:"
    echo "  bash scripts/build.sh siftsmall"
    echo "  bash scripts/deploy.sh siftsmall"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup complete with $WARNINGS warning(s)${NC}"
    echo ""
    echo "You can proceed, but some optional features may not work."
    echo "Review warnings above and install missing components if needed."
    exit 0
else
    echo -e "${RED}✗ Setup incomplete with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    echo "Refer to README.md for installation instructions."
    exit 1
fi
