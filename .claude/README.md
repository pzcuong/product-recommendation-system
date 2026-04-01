# 🚀 Scholarflow - Claude Code + GLM-4.7 Setup

## 📁 Configuration Structure

```
.claude/
├── CLAUDE.md              - GLM-4.7 coding guidelines
├── README.md              - This file
├── SETUP.md               - Detailed MCP setup guide
├── settings.json          - MCP servers & hooks config
├── settings.local.json    - Local settings
├── setup.sh               - Setup script
└── test.sh                - Test MCP servers
```

---

## ✅ What's Been Configured

### 1. **CLAUDE.md** - GLM-4.7 Guidelines

- Critical rules for GLM-4.7 stability
- File editing best practices (Read before Edit, relative paths)
- NestJS, Next.js, and Python AI/ML guidelines
- Project structure overview
- Common pitfalls to avoid

### 2. **Git Hooks Protection**

Auto-commit is enabled! Every file edit by GLM-4.7 will:

- Automatically create a checkpoint commit
- Allow easy rollback if something goes wrong
- Protect against bad edits

### 3. **MCP Servers**

Configured in `settings.json`:

- **GitHub MCP** - Create PRs, manage issues
- **Filesystem MCP** - Advanced file operations
- **Brave Search MCP** - Web search capabilities
- **Postgres MCP** - Database access

### 4. **Setup Scripts**

- `setup.sh` - Initial setup automation
- `test.sh` - Test MCP servers functionality

---

## 🚀 Quick Start

### Step 1: Run Setup

```bash
.claude/setup.sh
```

### Step 2: Configure MCP Servers

Edit `settings.json` and add your tokens directly:

```json
{
  "mcpServers": {
    "github": {
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"
      }
    }
  }
}
```

**Get GitHub Token:** https://github.com/settings/tokens

### Step 3: Restart Claude Code

Close and reopen Claude Code to load the new MCP servers.

### Step 4: Test Your Setup

```bash
.claude/test.sh
```

### Step 5: Try Claude Code with MCP

Ask Claude:

- "List all TypeScript files in the backend directory"
- "Create a test PR using GitHub MCP"
- "Show me the project structure"

---

## 🛡️ Git Hooks Protection

Your Git repository now has protection against GLM-4.7 errors:

**Auto-commit feature:**

- Every Edit/Write operation creates a checkpoint
- Commit message: `auto-checkpoint: [file changes]`

**Pre-commit hook:**

- Warns about committing .bak files
- Warns about GLM TODO comments

**To rollback a bad edit:**

```bash
# View recent checkpoints
git log --oneline -10

# Reset to previous checkpoint (keep changes)
git reset --soft HEAD~1

# Or completely discard changes
git reset --hard HEAD~1
```

---

## 📚 Documentation Files

| File            | Purpose                                       |
| --------------- | --------------------------------------------- |
| `CLAUDE.md`     | GLM-4.7 coding guidelines & project structure |
| `settings.json` | MCP servers & hooks configuration             |
| `SETUP.md`      | Detailed MCP setup guide                      |
| `setup.sh`      | Automated setup script                        |
| `test.sh`       | Test MCP servers functionality                |

---

## 🔍 Troubleshooting

### MCP Server Not Working

1. Check Node.js is installed: `node --version`
2. Check `settings.json` has correct tokens
3. Restart Claude Code
4. Run `.claude/test.sh` to diagnose

### Git Hooks Not Firing

1. Check `settings.json` has `hooks` section
2. Check git is initialized: `git status`
3. Re-run `.claude/setup.sh`

### GLM-4.7 Edit Errors

1. Always use relative paths (see `CLAUDE.md`)
2. Read file before editing
3. Make small, targeted edits
4. Check recent checkpoints: `git log --oneline -10`

---

## 🎯 Quick Reference

### Running the Project

```bash
# Backend
cd scholarflow-be && npm run start:dev

# Frontend
cd scholarflow-fe && npm run dev

# Python AI/ML
cd ace-main-1 && python -m ace.main
```

### Testing MCP Servers

```bash
.claude/test.sh
```

### Re-running Setup

```bash
.claude/setup.sh
```

---

## 🌐 External Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Code Skills](https://github.com/travisvn/awesome-claude-skills)
- [GLM-4.7 Troubleshooting](https://github.com/sgl-project/sglang/issues/15721)
- [Z.ai Vision MCP](https://blog.devgenius.io/fixing-glm-4-7-image-parsing-in-claude-code-add-the-z-ai-vision-mcp-server-f1c275d7cf3f)

---

## 🎉 Features

Your Scholarflow project is now optimized for:

- ✅ GLM-4.7 stability
- ✅ Git protection & easy rollback
- ✅ MCP servers for enhanced capabilities
- ✅ Best practices documentation

**Happy coding with Claude Code + GLM-4.7! 🚀**

---

_Scholarflow - AI-Powered Learning Platform_
_Optimized for Claude Code with GLM-4.7_
