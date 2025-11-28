# OpenSpec Commands for Cursor

This file defines slash commands for OpenSpec integration with Cursor.

## Available Commands

### /openspec:proposal
Create a new OpenSpec change proposal.

**Usage**: `/openspec:proposal <description>`

**Example**: `/openspec:proposal Add brain tumor detection pipeline`

**What it does**:
1. Creates a new change folder in `openspec/changes/`
2. Generates `proposal.md` with problem statement and solution
3. Creates `tasks.md` with implementation checklist
4. Sets up `specs/` directory for spec deltas

### /openspec:apply
Apply (implement) an OpenSpec change.

**Usage**: `/openspec:apply <change-name>`

**Example**: `/openspec:apply add-brain-tumor-detection`

**What it does**:
1. Reads tasks from `openspec/changes/<change-name>/tasks.md`
2. Implements each task according to spec deltas
3. Marks completed tasks with `[x]`
4. Follows project conventions and style guides

### /openspec:archive
Archive a completed OpenSpec change.

**Usage**: `/openspec:archive <change-name>`

**Example**: `/openspec:archive add-brain-tumor-detection`

**What it does**:
1. Executes `openspec archive <change-name> --yes`
2. Moves change to `openspec/archive/`
3. Merges spec deltas into `openspec/specs/`
4. Confirms successful archival

### /openspec:show
Display details of an OpenSpec change.

**Usage**: `/openspec:show <change-name>`

**Example**: `/openspec:show add-brain-tumor-detection`

**What it does**:
1. Shows the proposal content
2. Lists all tasks and their status
3. Displays relevant spec deltas

### /openspec:list
List all active OpenSpec changes.

**Usage**: `/openspec:list`

**What it does**:
1. Executes `openspec list`
2. Displays all changes in `openspec/changes/`

### /openspec:validate
Validate an OpenSpec change's spec format.

**Usage**: `/openspec:validate <change-name>`

**Example**: `/openspec:validate add-brain-tumor-detection`

**What it does**:
1. Executes `openspec validate <change-name>`
2. Checks spec delta formatting
3. Reports any validation errors

## Command Workflow

### Typical workflow for a new feature:

1. **Create proposal**:
   ```
   /openspec:proposal Add MRI T2-FLAIR sequence support
   ```

2. **Review and refine**:
   ```
   /openspec:show add-mri-t2-flair-support
   ```
   Manually edit `proposal.md`, `tasks.md`, and spec deltas as needed.

3. **Validate specs**:
   ```
   /openspec:validate add-mri-t2-flair-support
   ```

4. **Implement**:
   ```
   /openspec:apply add-mri-t2-flair-support
   ```

5. **Archive when complete**:
   ```
   /openspec:archive add-mri-t2-flair-support
   ```

## Notes for AI Assistant

- Always read `AGENTS.md` for full OpenSpec workflow guidance
- Follow project conventions in `openspec/project.md`
- Respect Linus style rules (see `.cursor/rules/linus.mdc`)
- For medical imaging projects, include clinical validation criteria
- Use proper DICOM terminology and safety considerations
- Maintain backward compatibility with existing pipelines

