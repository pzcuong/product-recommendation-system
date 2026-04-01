#!/bin/bash

# Scholarflow - MCP Servers Test Script
# This script tests all configured MCP servers

set -e

echo "🧪 Testing Scholarflow MCP Servers..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counter
passed=0
failed=0

# Test function
test_server() {
    local name=$1
    local command=$2

    echo -e "${BLUE}Testing $name...${NC}"

    if eval $command > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name is working${NC}"
        ((passed++))
    else
        echo -e "${RED}✗ $name test failed${NC}"
        ((failed++))
    fi
    echo ""
}

# Test 1: Node.js and NPM
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Testing Prerequisites${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if command -v node &> /dev/null; then
    echo -e "${GREEN}✓ Node.js $(node --version)${NC}"
else
    echo -e "${RED}✗ Node.js not found${NC}"
    ((failed++))
fi

if command -v npm &> /dev/null; then
    echo -e "${GREEN}✓ npm $(npm --version)${NC}"
else
    echo -e "${RED}✗ npm not found${NC}"
    ((failed++))
fi

echo ""

# Test 2: Claude Code Configuration
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Testing Claude Code Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/settings.json" ]; then
    echo -e "${GREEN}✓ .claude/settings.json exists${NC}"
else
    echo -e "${RED}✗ .claude/settings.json not found${NC}"
    ((failed++))
fi

if [ -f "$SCRIPT_DIR/CLAUDE.md" ]; then
    echo -e "${GREEN}✓ .claude/CLAUDE.md exists${NC}"
else
    echo -e "${YELLOW}⚠ .claude/CLAUDE.md not found (recommended)${NC}"
fi

echo ""

# Test 3: MCP Server Installation
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Testing MCP Server Installation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

test_server "GitHub MCP" "npx -y @modelcontextprotocol/server-github --version"
test_server "Filesystem MCP" "npx -y @modelcontextprotocol/server-filesystem --version"
test_server "Brave Search MCP" "npx -y @modelcontextprotocol/server-brave-search --version"
test_server "Postgres MCP" "npx -y @modelcontextprotocol/server-postgres --version"

echo ""

# Test 5: Git Repository
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Testing Git Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Git repository initialized${NC}"

    if [ -f .git/hooks/pre-commit ]; then
        echo -e "${GREEN}✓ Git pre-commit hook installed${NC}"
    else
        echo -e "${YELLOW}⚠ Git pre-commit hook not installed${NC}"
    fi

    current_branch=$(git branch --show-current)
    echo -e "${BLUE}Current branch: $current_branch${NC}"
else
    echo -e "${YELLOW}⚠ Not a git repository${NC}"
fi

echo ""

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Passed: $passed${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Your setup is ready.${NC}"
    echo ""
    echo "🎉 You can now use Claude Code with MCP servers!"
    echo ""
    echo "Try these commands:"
    echo "  • 'List all TypeScript files in the backend'"
    echo "  • 'Create a test PR using GitHub MCP'"
    echo "  • 'Search the web for GLM-4.7 best practices'"
else
    echo -e "${YELLOW}⚠ Some tests failed. Please check the errors above.${NC}"
    echo ""
    echo "Fix steps:"
    echo "  1. Make sure Node.js and npm are installed"
    echo "  2. Run: .claude/setup.sh"
    echo "  3. Configure MCP servers in .claude/settings.json"
    echo "  4. Restart Claude Code"
fi
echo ""
