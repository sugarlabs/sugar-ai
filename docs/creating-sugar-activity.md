#  Creating Sugar Activities — A Beginner-Friendly Guide

Welcome to the ultimate guide on **Creating Sugar Activities** — crafted for developers, educators, and curious minds who want to build interactive educational apps for the Sugar Learning Platform. Whether you're a student trying your first open-source contribution or a developer exploring Sugar Labs, this guide will help you create and run your own Sugar activity from scratch — step-by-step.

---

## What Is a Sugar Activity?

A **Sugar Activity** is a Python-based educational application that runs within the [Sugar Learning Platform](https://sugarlabs.org/). It's designed especially for children, offering a simple, colorful, and distraction-free interface that promotes learning by doing.

Each activity runs in its own sandboxed environment, is open-source, and supports collaboration — a perfect tool for teaching, creating, and contributing.

---

## Prerequisites

Before you dive in, ensure you have the following set up:

- Python (preferably Python 3)
- Linux (Fedora is preferred; Ubuntu/Debian also works)
- [Sugar environment](https://github.com/sugarlabs/sugar)
- `sugar-toolkit-gtk3` installed
- Git and basic terminal knowledge

You can also use a Sugar emulator or try activities directly on a [Sugar on a Stick (SoaS)](https://wiki.sugarlabs.org/go/Sugar_on_a_Stick).

---

##  Basic Structure of a Sugar Activity

Here’s a glance at what your activity directory will look like:

```
MyActivity/
├── activity/
│   └── activity.info        # Metadata file
├── MyActivity.py            # Main logic
├── icons/
│   └── my-icon.svg          # SVG icon for the activity
├── MANIFEST                 # List of files to package
├── setup.py (optional)     # For advanced builds
```

### Explanation:
- `activity.info` — A required file with details like name, bundle_id, exec, icon, and activity_version.
- `MyActivity.py` — This is your main application file.
- `icons/` — Keep your activity's visual identity here.
- `MANIFEST` — Lists the files to be included in the .xo package.

---

##  Step-by-Step: Create Your First Activity

### Step 1: Clone the Template

Sugar Labs has a helpful starter repo:
git clone https://github.com/sugarlabs/make-your-own-sugar-activities.git
cd make-your-own-sugar-activities/helloworld


###  Step 2: Explore and Understand
Check out the `activity.info` file and `helloworld.py`. This activity shows a button that prints "Hello World" when clicked.

###  Step 3: Run the Activity
You can run Sugar and test the activity using:

sugar-emulator
Or package it as `.xo`:

./setup.py dist_xo
Install with:
sugar-install-bundle HelloWorld-1.xo


###  Step 4: Customize
Rename the folder and files, update the metadata in `activity.info`, and start building your own educational logic!

---

##  Tools That Help

- **sugar-toolkit-gtk3**: Essential for working with the Sugar environment.
- **Pippy**: A good example to see working Python code.
- **Write Activity**: Advanced example showing save/load and collaboration.

---

##  Sugar AI Context (Why This Is Needed)

Sugar AI wants to make the development of Sugar Activities even easier by:
- Explaining what Sugar Activities are
- Providing tools and examples via a knowledge base
- Supporting AI-powered suggestions for new activities

If you’re working on `sugar-ai`, this documentation should be linked and referenced in the README or docs folder.

---

##  Minimal Example Code

 ''' python 
from sugar3.activity import activity
from gi.repository import Gtk

class HelloWorldActivity(activity.Activity):
    def __init__(self, handle):
        activity.Activity.__init__(self, handle)

        self.set_title("Hello World")
        button = Gtk.Button(label="Click me!")
        button.connect("clicked", self.on_click)
        self.set_canvas(button)
        button.show()

    def on_click(self, button):
        print("Hello, Sugar World!")
```

---

## 📎 Reference Links

-  [Make Your Own Sugar Activities](https://github.com/sugarlabs/make-your-own-sugar-activities)
-  [Sugar Developer Portal](https://developer.sugarlabs.org)
-  [Sugar Toolkit GTK3](https://github.com/sugarlabs/sugar-toolkit-gtk3)
-  [Pippy Activity](https://github.com/sugarlabs/pippy)

---

##  Final Words

By contributing this documentation, you’re helping lower the barrier for new contributors to understand and build for Sugar. If you’re including this in a GitHub PR for `sugar-ai`, link this in a `docs/` folder or integrate it into the main README under a new `Creating Activities` section.

Let’s keep making education open and fun!

---

 *For questions, suggestions, or feedback, feel free to reach out on [Sugar Labs IRC](https://wiki.sugarlabs.org/go/IRC) or [GitHub Discussions](https://github.com/sugarlabs/sugar/discussions).*


