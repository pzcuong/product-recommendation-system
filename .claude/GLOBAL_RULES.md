# GLOBAL_RULES.md - Experience Lessons

## 🧠 Core Mindset: Verify → Analyze → Fix

Most bugs come from wrong assumptions. Before fixing:

1. **VERIFY** - What does the data/error actually look like?
2. **ANALYZE** - Why is it happening? Trace the flow.
3. **FIX** - Address the root cause, not the symptom.

---

## Rule 1: Trust Logs, Not Assumptions

When you encounter an error, the first thing to check is the actual error structure.

**Mistake:** I assumed `error.response.data.code` was the path based on common patterns.
**Reality:** The actual structure was `error.response.code` - I should have logged it first.

**Takeaway:** Always `JSON.stringify(error)` or `console.log(error)` before writing error handling code. The actual structure might surprise you.

---

## Rule 2: Compare Side-by-Side

When similar code behaves differently (works in one place, fails in another), compare them directly.

**Mistake:** I fixed the list query and detail query separately, assuming they were independent.
**Reality:** List used INNER JOIN (required modules), detail used findOne (no module check). This mismatch caused the bug.

**Takeaway:** Open both files/functions side by side. What condition/filter is different? Why does one include items the other excludes?

---

## Rule 3: Check the Source of Truth

When debugging data issues, go to the entity definition or database directly.

**Mistake:** I used `.innerJoin('course_modules', ...)` assuming that was the table name.
**Reality:** TypeORM needs the relation name `course.modules`, not the table name.

**Takeaway:** Before using any relation/join, check the entity definition. The property name in `@OneToMany()` is what TypeORM expects.

---

## Rule 4: Understand the "Why"

Before fixing a symptom, ask why the data is structured that way.

**Mistake:** I set `id: null` for roadmaps without courses, thinking this fixed the issue.
**Reality:** The real question was: Why are these roadmaps being returned at all? They shouldn't be in the query results.

**Takeaway:** "Why is this data here?" is more important than "How do I format this data?" Filter at the source, not in the mapping layer.

---

## Rule 5: Order Matters in Data Processing

Pagination + filtering order is a common source of bugs.

**Mistake:** I paginated first, then filtered. The metadata said 24 items, but the array was empty.
**Reality:** Pagination happened on the unfiltered data. The filter removed everything after.

**Takeaway:** If pagination metadata doesn't match array length, the filter is applied too late. Move filters into the query's WHERE clause.

---

## Rule 6: Follow the Transaction Path

When you see orphaned or partial data, trace the creation flow.

**Mistake:** I tried to filter out roadmaps without courses from queries.
**Reality:** These roadmaps were being created BEFORE credit deduction. When credits failed, roadmaps were orphaned.

**Takeaway:** Partial data usually means a transaction failed mid-flow. Find where the failure point is and either (a) move the validation earlier or (b) add proper rollback/cleanup.

---

## Rule 7: Query the Database Directly

When you're stuck, stop coding and check what's actually in the database.

**Mistake:** I kept adding filters to queries assuming what the data looked like.
**Reality:** A simple `SELECT * FROM courses WHERE id = '...'` would have shown the course existed but had no modules.

**Takeaway:** Five minutes of direct database investigation saves hours of wild guessing. Use `psql`, database GUI, or add debug logging to see the actual data.

---

## 🎯 Red Flags: You're Probably On the Wrong Track

- **Adding more filters** without understanding why the existing ones don't work
- **Changing multiple files** without knowing which one is actually the problem
- **Copying patterns** from other parts of the code without understanding why they work there
- **Fixing FE** when the issue is clearly in BE (or vice versa)
- **Wrapping errors** in try-catch without re-throwing specific ones the FE needs
- **Assuming data structure** instead of checking the entity definition or database

---

## 💡 Universal Debugging Workflow

1. **See the actual error/data** - Log it, read it carefully
2. **Trace the full flow** - From database → service → controller → frontend
3. **Compare working vs broken** - What's different between them?
4. **Verify at the source** - Check the database, entity definitions, actual API responses
5. **Fix the root cause** - Not just the symptom

---

## 📌 Remember

Every bug reveals something about how the system **actually works** versus how you **think it works**.

The difference between your mental model and reality is where bugs live.
