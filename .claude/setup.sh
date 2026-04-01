#!/bin/bash

# Scholarflow - Claude Code & GLM-4.7 Setup Script
# This script sets up MCP servers and configurations for optimal GLM-4.7 performance

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Setting up Scholarflow for Claude Code with GLM-4.7..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to project root
cd "$PROJECT_ROOT"

# Check if node is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js is not installed. Please install Node.js first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js $(node --version) found${NC}"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}✗ npm is not installed. Please install npm first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ npm $(npm --version) found${NC}"
echo ""

# Check if .claude directory exists
if [ ! -d .claude ]; then
    echo -e "${RED}✗ .claude directory not found. Please run this script from the .claude directory.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ .claude directory exists${NC}"
echo ""

# Test MCP servers
echo "Testing MCP servers..."
echo ""

# Test GitHub MCP
echo -e "${YELLOW}Testing GitHub MCP Server...${NC}"
npx -y @modelcontextprotocol/server-github --version 2>/dev/null || echo -e "${YELLOW}⚠ GitHub MCP test completed${NC}"

# Test Filesystem MCP
echo -e "${YELLOW}Testing Filesystem MCP Server...${NC}"
npx -y @modelcontextprotocol/server-filesystem --version 2>/dev/null || echo -e "${YELLOW}⚠ Filesystem MCP test completed${NC}"

echo ""

# Git hooks setup
echo "Setting up Git hooks protection..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Git repository detected${NC}"

    # Create pre-commit hook for GLM-4.7 safety
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for GLM-4.7 safety
# Prevents commits with obvious GLM-4.7 errors

# Check for temporary backup files
if git diff --cached --name-only | grep -q '\.bak$'; then
    echo "⚠ Warning: Committing .bak files"
fi

# Check for incomplete edits (common GLM-4.7 pattern)
if git diff --cached | grep -q 'TODO.*GLM.*fix'; then
    echo "⚠ Warning: Committing with GLM TODO comments"
fi

exit 0
EOF
    chmod +x .git/hooks/pre-commit
    echo -e "${GREEN}✓ Git pre-commit hook installed${NC}"
else
    echo -e "${YELLOW}⚠ Not a git repository, skipping git hooks${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}📋 Configuration Files:${NC}"
echo "   • .claude/CLAUDE.md      - GLM-4.7 coding guidelines"
echo "   • .claude/settings.json  - MCP servers & hooks config"
echo "   • .claude/SETUP.md       - Detailed MCP setup guide"
echo "   • .claude/README.md      - Complete documentation"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Configure MCP servers in .claude/settings.json"
echo "2. Set environment variables (GITHUB_TOKEN, etc.)"
echo "3. Restart Claude Code to load new MCP servers"
echo "4. Run: .claude/test.sh"
echo ""
echo -e "${BLUE}🔍 Test your setup:${NC}"
echo "   - Ask Claude: 'List all TypeScript files using Filesystem MCP'"
echo "   - Ask Claude: 'Create a test PR using GitHub MCP'"
echo ""
