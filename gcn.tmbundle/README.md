# GCN TextMate/VS Code Language Support

Syntax highlighting for GCN (gEconpy Configuration Notation) files used by the [gEconpy](https://github.com/jessegrabowski/gEconpy) DSGE modeling library.

## Features

- **Keywords**: `block`, `options`, `tryreduce`, `assumptions`
- **Components**: `definitions`, `controls`, `objective`, `constraints`, `identities`, `shocks`, `calibration`
- **Assumptions**: `positive`, `negative`, `nonnegative`, `nonpositive`, `real`, `integer`, `finite`
- **Variables with time indices**: `C[]`, `K[-1]`, `Y[ss]`
- **Expectation operator**: `E[][...]`
- **Lagrange multipliers**: `: lambda[]`
- **Calibrating parameters**: `-> param`
- **Tags**: `@exclude`
- **PreliZ distributions**: `Normal`, `Beta`, `Gamma`, etc.
- **Distribution wrappers**: `maxent`, `Censored`, `Truncated`, `Hurdle`
- **Mathematical functions**: `log`, `exp`, `sqrt`, etc.
- **Comments**: `# comment`

## Installation

### VS Code

1. Copy this folder to your VS Code extensions directory:
   - **macOS**: `~/.vscode/extensions/gcn.tmbundle`
   - **Linux**: `~/.vscode/extensions/gcn.tmbundle`
   - **Windows**: `%USERPROFILE%\.vscode\extensions\gcn.tmbundle`

2. Restart VS Code

### TextMate (macOS)

1. Double-click the `gcn.tmbundle` folder, or
2. Copy to `~/Library/Application Support/TextMate/Bundles/`

### Sublime Text

Copy the `Syntaxes/gcn.tmLanguage.json` file to your Sublime Text packages folder.

### PyCharm / IntelliJ IDEA

1. Go to **Settings/Preferences → Editor → TextMate Bundles**
2. Click the **+** button and select this `gcn.tmbundle` folder
3. Click **Apply** and restart the IDE

Alternatively, copy the bundle to:
- **macOS**: `~/Library/Application Support/JetBrains/<product>/textmate/`
- **Linux**: `~/.config/JetBrains/<product>/textmate/`
- **Windows**: `%APPDATA%\JetBrains\<product>\textmate\`

Where `<product>` is your IDE version folder (e.g., `PyCharm2025.1`).


## Troubleshooting

### Variables not highlighting in PyCharm

If variables like `C[]`, `K[-1]`, `Y[ss]` are not highlighting:

1. **Restart PyCharm completely** (not just reload the bundle)
2. **Invalidate caches**: Use **File → Invalidate Caches → Invalidate and Restart**
3. **Re-add the bundle**: Remove and re-add via **Settings → Editor → TextMate Bundles**
4. **Check scope assignment**: Place the caret on a variable and use:
   - **Edit → Find → Find Action** → type "Show Tokens" or "TextMate Scopes"
   - Or check **Settings → Editor → Color Scheme → TextMate** to verify scopes have colors
5. **Verify file association**: Ensure `.gcn` files are associated with the GCN file type under **Settings → Editor → File Types**

### Color scheme doesn't show all scopes

Some color schemes may not have colors assigned for all TextMate scopes. The grammar uses:
- `variable.other.readwrite.gcn` - Variables with time indices
- `variable.parameter.gcn` - Parameters (plain identifiers)
- `keyword.control.block.gcn` - Block keyword
- `entity.name.type.block.gcn` - Block names
- `support.type.distribution.gcn` - Distribution names

You can customize these in **Settings → Editor → Color Scheme → TextMate**.

## Example GCN File

```gcn
assumptions
{
    positive { C[], K[], L[], Y[]; };
};

tryreduce { U[]; };

block HOUSEHOLD
{
    definitions
    {
        u[] = C[] ^ (1 - gamma) / (1 - gamma);
    };

    controls
    {
        C[], K[];
    };

    objective
    {
        U[] = u[] + beta * E[][U[1]];
    };

    constraints
    {
        C[] + K[] = r[] * K[-1] + w[] * L[] : lambda[];
    };

    calibration
    {
        beta ~ maxent(Beta(), lower=0.95, upper=0.999) = 0.99;
        gamma ~ Normal(mu=2, sigma=0.5) = 2.0;
    };
};
```
