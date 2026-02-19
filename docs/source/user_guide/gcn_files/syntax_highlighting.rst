Syntax Highlighting
===================

gEconpy includes a TextMate grammar bundle that provides syntax highlighting for GCN files in most modern editors. The bundle is located in the ``gcn.tmbundle`` directory in the gEconpy repository.


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
