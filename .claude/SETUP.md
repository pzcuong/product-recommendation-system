# Claude Code MCP Servers Setup Guide

## 📋 Overview

This guide helps you set up the essential MCP (Model Context Protocol) servers for Scholarflow project with GLM-4.7.

---

## 🔧 Prerequisites

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# GitHub Token (for GitHub MCP Server)
GITHUB_TOKEN=your_github_personal_access_token_here

# Brave Search API Key (optional - for web search)
BRAVE_API_KEY=your_brave_api_key_here

# Database Connection (update with your actual credentials)
DATABASE_URL=postgresql://postgres:password@localhost:5432/scholarflow
```

### Getting GitHub Token:

1. Go to https://github.com/settings/tokens
2. Generate new token → Classic
3. Select scopes: `repo`, `read:org`
4. Copy and paste to `.env`

### Getting Brave API Key (Optional):

1. Go to https://api.search.brave.com/app/keys
2. Sign up for free API access
3. Copy and paste to `.env`

---

## 🚀 Quick Install

Run the setup script:

```bash
npm run setup:mcp
```

Or manually install each MCP server:

```bash
# GitHub MCP Server
npx -y @modelcontextprotocol/server-github

# Filesystem MCP Server
npx -y @modelcontextprotocol/server-filesystem /Users/macbook/Desktop/scholarflow

# Brave Search MCP Server (optional)
npx -y @modelcontextprotocol/server-brave-search

# Postgres MCP Server
npx -y @modelcontextprotocol/server-postgres "postgresql://postgres:password@localhost:5432/scholarflow"
```

---

## 📦 MCP Servers Included

### 1. **GitHub MCP Server** ⭐ (Essential)

- **Purpose**: Create PRs, review code, manage issues
- **Config**: `.claude/settings.json`
- **Required**: `GITHUB_TOKEN` env variable

### 2. **Filesystem MCP Server** ⭐ (Essential)

- **Purpose**: Advanced file operations, search, watching
- **Config**: Already configured for `/Users/macbook/Desktop/scholarflow`

### 3. **Brave Search MCP Server** (Optional)

- **Purpose**: Web search capabilities
- **Config**: Requires `BRAVE_API_KEY`

### 4. **Postgres MCP Server** (Optional)

- **Purpose**: Direct database access
- **Config**: Update connection string in `settings.json`

---

## 🛡️ Git Hooks Protection

Auto-commit is enabled! Every time GLM-4.7 edits or writes a file, it will:

1. Automatically `git add -A`
2. Create a checkpoint commit with message: `auto-checkpoint: [file changes]`

### To disable Git Hooks:

Edit `.claude/settings.json` and remove the `hooks` section.

### To rollback a bad edit:

```bash
# View recent auto-checkpoints
git log --oneline -10

# Reset to previous checkpoint
git reset --soft HEAD~1

# Or completely discard changes
git reset --hard HEAD~1
```

---

## 🧪 Test MCP Servers

After setup, test with Claude Code:

```
Can you create a test PR using the GitHub MCP server?
```

Or check file watching:

```
List all TypeScript files in the backend directory
```

---

## 🔍 Troubleshooting

### MCP Server Not Working:

1. Check if Node.js is installed: `node --version`
2. Check if package is installed: `npx -y @modelcontextprotocol/server-[name]`
3. Check `.claude/settings.json` syntax
4. Check environment variables are set

### Git Hooks Not Firing:

1. Check `.claude/settings.json` has `hooks` section
2. Check permissions include git commands
3. Check git is initialized: `git status`

### Z.ai Vision MCP Server (Advanced):

For GLM-4.7 image parsing issues, install the Z.ai Vision MCP server:

```bash
# Follow guide: https://blog.devgenius.io/fixing-glm-4-7-image-parsing-in-claude-code-add-the-z-ai-vision-mcp-server-f1c275d7cf3f
```

---

## 📚 Additional Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Code Skills](https://github.com/travisvn/awesome-claude-skills)
- [GLM-4.7 Troubleshooting](https://github.com/sgl-project/sglang/issues/15721)

---

_Generated for Scholarflow - AI-Powered Learning Platform_
