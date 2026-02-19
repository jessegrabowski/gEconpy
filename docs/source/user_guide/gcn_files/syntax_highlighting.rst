Syntax Highlighting
===================

gEconpy includes a TextMate grammar bundle that provides syntax highlighting for GCN files in most modern editors. The bundle is located in the ``gcn.tmbundle`` directory in the gEconpy repository.

Features
--------

The grammar provides highlighting for:

- **Block keywords**: ``block``, ``options``, ``tryreduce``, ``assumptions``
- **Block components**: ``definitions``, ``controls``, ``objective``, ``constraints``, ``identities``, ``shocks``, ``calibration``
- **Assumption keywords**: ``positive``, ``negative``, ``nonnegative``, ``nonpositive``, ``real``, ``integer``, ``finite``
- **Variables with time indices**: ``C[]``, ``K[-1]``, ``Y[ss]``
- **Expectation operator**: ``E[][...]``
- **Lagrange multipliers**: ``: lambda[]``
- **Calibration targets**: ``-> param``
- **Tags**: ``@exclude``
- **Prior distributions**: ``Normal``, ``Beta``, ``Gamma``, etc.
- **Distribution wrappers**: ``maxent``, ``Censored``, ``Truncated``, ``Hurdle``
- **Mathematical functions**: ``log``, ``exp``, ``sqrt``, etc.
- **Comments**: ``# comment``


Installation
------------

VS Code
^^^^^^^

1. Copy the ``gcn.tmbundle`` folder to your VS Code extensions directory:

   - **macOS**: ``~/.vscode/extensions/gcn.tmbundle``
   - **Linux**: ``~/.vscode/extensions/gcn.tmbundle``
   - **Windows**: ``%USERPROFILE%\.vscode\extensions\gcn.tmbundle``

2. Restart VS Code


PyCharm / IntelliJ IDEA
^^^^^^^^^^^^^^^^^^^^^^^

1. Go to **Settings/Preferences → Editor → TextMate Bundles**
2. Click the **+** button and select the ``gcn.tmbundle`` folder
3. Click **Apply** and restart the IDE

Alternatively, copy the bundle to the IDE's TextMate directory:

- **macOS**: ``~/Library/Application Support/JetBrains/<product>/textmate/``
- **Linux**: ``~/.config/JetBrains/<product>/textmate/``
- **Windows**: ``%APPDATA%\JetBrains\<product>\textmate\``

Where ``<product>`` is your IDE version folder (e.g., ``PyCharm2025.1``).


TextMate (macOS)
^^^^^^^^^^^^^^^^

Double-click the ``gcn.tmbundle`` folder, or copy it to:

``~/Library/Application Support/TextMate/Bundles/``


Sublime Text
^^^^^^^^^^^^

Copy ``gcn.tmbundle/Syntaxes/gcn.tmLanguage.json`` to your Sublime Text packages folder:

- **macOS**: ``~/Library/Application Support/Sublime Text/Packages/User/``
- **Linux**: ``~/.config/sublime-text/Packages/User/``
- **Windows**: ``%APPDATA%\Sublime Text\Packages\User\``


Troubleshooting
---------------

Variables not highlighting in PyCharm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If variables like ``C[]``, ``K[-1]``, ``Y[ss]`` are not highlighting:

1. **Restart PyCharm completely** (not just reload the bundle)
2. **Invalidate caches**: Use **File → Invalidate Caches → Invalidate and Restart**
3. **Re-add the bundle**: Remove and re-add via **Settings → Editor → TextMate Bundles**
4. **Verify file association**: Ensure ``.gcn`` files are associated with the GCN file type under **Settings → Editor → File Types**


Color scheme doesn't show all scopes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some color schemes may not have colors assigned for all TextMate scopes. You can customize scope colors in:

- **PyCharm**: **Settings → Editor → Color Scheme → TextMate**
- **VS Code**: Edit your ``settings.json`` and add ``editor.tokenColorCustomizations``
