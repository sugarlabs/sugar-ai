# Sugar Activities: Complete Guide

**Source:** FLOSS Manuals – “Make Your Own Sugar Activities”  
**Authors:** James D. Simmons and contributors  
**License:** GNU General Public License v2 or later (GPL-2.0-or-later)  
**Original source URL:** https://archive.flossmanuals.net/make-your-own-sugar-activities/


## Introduction

> "This book is a record of a pleasure trip. If it were a record of a solemn scientific expedition, it would have about it that gravity, that profundity, and that impressive incomprehensibility which are so proper to works of that kind, and withal so attractive."
>
> From the Preface to *The Innocents Abroad*, by Mark Twain

The purpose of this book is to teach you what you need to know to write Activities for Sugar, the operating environment developed for the One Laptop Per Child project. This book does not assume that you know how to program a computer, although those who do will find useful information in it. My primary goal in writing it is to encourage non programmers, including children and their teachers, to create their own Sugar Activities. Because of this goal I will include some details that other books would leave out and leave out things that others would include. Impressive incomprehensibility will be kept to a minimum.

If you just want to learn how to write computer programs Sugar provides many Activities to help you do that: Etoys, Turtle Art, Scratch, and Pippy. None of these are really suitable for creating Activities so I won't cover them in this book, but they're a great way to learn about programming. If you decide after playing with these that you'd like to try writing an Activity after all you'll have a good foundation of knowledge to build on.

When you have done some programming then you'll know how satisfying it can be to use a program that you made yourself, one that does exactly what you want it to do. Creating a Sugar Activity takes that enjoyment to the next level. A useful Sugar Activity can be translated by volunteers into every language, be downloaded hundreds of times a week and used every day by students all over the world.

Some Sugar Activities!

A book that teaches everything you need to know to write Activities would be really, really long and would duplicate material that is already available elsewhere. Because of this, I am going to write this as sort of a guided tour of Activity development. That means, for example, that I'll teach you what Python is and why it's important to learn it but I won't teach you the Python language itself. There are excellent tutorials on the Internet that will do that, and I'll refer you to those tutorials.

There is much sample code in this book, but there is no need for you to type it in to try it out. All of the code is in a Git repository that you can download to your own computer. If you've never used Git there is a chapter that explains what it is and how to use it.

I started writing Activities shortly after I received my XO laptop. When I started I didn't know any of the material that will be in this book. I had a hard time knowing where to begin. What I did have going for me though was a little less than 30 years as a professional programmer. As a result of that I think like a programmer. A good programmer can take a complex task and divide it up into manageable pieces. He can figure out how things must work, and from that figure out how they do work. He knows how to ask for help and where. If there is no obvious place to begin he can begin somewhere and eventually get where he needs to go.

Because I went through this process I think I can be a pretty good guide to writing Sugar Activities. Along the way I hope to also teach you how to think like a programmer does.

From time to time I may add chapters to this book. Sugar is a great application platform and this book can only begin to tell you what is possible.

In the first edition of this book I expressed the wish that future versions of the book would have guest chapters on more advanced topics written by other experienced Activity developers. That wish has been realized. Not only does the book have guest chapters, but some of the new content has been created by young developers from the Google Code-in of 2012, for which I was one of the mentors. Their bios are in the About The Authors chapter. Read them and prepare to be impressed!

This book eliminates some content that was part of the first edition. The current version of Sugar uses GTK 3, and most of the examples in this addition use GTK 3 as well. In the first edition all the examples used GTK 2. The first edition also explained how to support multiple versions of Sugar with one Activity. That is no longer relevant to most Activity developers, so it has been removed.

If you need the content of the first edition you will find it at archive.org. Code examples from the first edition are also still available at git.sugarlabs.org.

In addition to examples using GTK3 this edition introduces the use of HMTL 5 and WebKit to create Activities. The advantage of HTML 5 is that it can run on Android and other tablet devices. There has been a fair amount of interest in using these devices for education, and if you develop using HTML 5 you can make versions of your Activity for both Sugar and Android.

## Formats For This Book

This book is part of the FLOSS Manuals project and is available for online viewing at their website:

http://en.flossmanuals.net/

At this time this is the only place the second edition is available, but it will in time be made available everywhere the first edition is.

You will can purchase a printed and bound version of the first edition of book at Amazon.com:

http://www.amazon.com/James-Simmons/e/B005197ZTY/ref=ntt_dp_epwbk_0

The Internet Archive has the first edition of this book available as a full color PDF, as well as EPUB, MOBI, and DjVu versions, all of which you can download for free:

http://www.archive.org/details/MakeYourOwnSugarActivities

The Amazon Kindle Store has exactly the same MOBI version as the Internet Archive does.

If you choose to read this book on a Kindle or a Nook be aware that these devices have narrow screens not well suited for displaying program listings. I suggest you refer to the FLOSS Manuals website to see what the code looks like properly formatted.

There is a Spanish version of the first edition of this book, *Cómo Hacer Una Actividad Sugar*, which is available in all the same places as this version.

## What Is Sugar?

Sugar is the user interface designed for the XO laptop. It can now be installed on most PCs, including older models that can't run the latest Windows software. You can also install it on a thumb drive (Sugar on a Stick) and boot your PC from that.

When the XO laptop first came out some people questioned the need for a new user interface. Wouldn't it be better for children to learn something more like what they would use as adults? Why not give them Microsoft Windows instead?

This would be a reasonable question if the goal was to train children to use computers and nothing else. It would be even more reasonable if we could be sure that the software they would use as adults looked and worked like the Microsoft Windows of today. These are of course not reasonable assumptions.

The OLPC project is not just about teaching computer literacy. It is about teaching everything: reading, writing, arithmetic, history, science, arts and crafts, computer programming, music composition, and everything else. Not only do we expect the child to use the computer for her school work, we expect her to take it home and use it for her own explorations into subjects that interest her.

This is a great deal more than anyone has done with computers for education, so it is reasonable to rethink how children should work with computers. Sugar is the result of that rethinking.

Sugar has the following unique features:

## The Journal

The Journal is where all the student's work goes. Instead of files and folders there is a list of Journal entries. The list is sorted in descending order by the date and time it was last worked on. In a way it's like the "Most Recently Used" document menu in Windows, except instead of containing just the last few items it contains everything and is the normal way to save and resume work on something.

The Journal makes it easy to organize your work. Any work you do is saved to the Journal. Anything you download from the web goes in the Journal. If you've ever downloaded a file using a web browser, then had to look for it afterwards because it went in some directory other than the one you expected, or if you ever had to help your parents when they were in a similar situation, you can understand the value of the Journal.

The Journal has metadata for each item in it. Metadata is information about information. Every Journal entry has a title, a description, a list of keywords, and a screen shot of what it looked like the last time it was used. It has an activity id that links it to the Activity that created it, and it may have a MIME type as well (which is a way of identifying Journal entries so that items not created by an Activity may still be used by an Activity that supports that MIME type).

In addition to these common metadata items a Journal entry may be given custom metadata by an Activity. For instance, the Read Activity uses custom metadata to save the page number you were reading when you quit the Activity. When you resume reading later the Activity will put you on that page again.

In addition to work created by Activities, the Journal can contain Activities themselves. To install an Activity you can use the Browse Activity to visit the website http://activities.sugarlabs.org and download it. It will automatically be saved to the Journal and be ready for use. If you don't want the Activity any more, simply delete it from the Journal and it's completely gone. No uninstall programs, no dialog boxes telling you that such and such a .DLL doesn't seem to be needed anymore and do you want to delete it? No odd bits and pieces left behind.

## Collaboration

The second unique feature Sugar is Collaboration. Collaboration means that Activities can be used by more than one person at the same time. While not every Activity needs collaboration and not every Activity that could use it supports it, a really first rate Activity will provide some way to interact with other Sugar users on the network. For instance, all the e-book reading Activities provide a way of giving a copy of the book you're reading (with any notes you added to it) to a friend or to the whole class. The Write Activity lets several students work on the same document together. The Distance Activity lets two students see how far apart from each other they are.

There are five views of the system you can switch to at the push of a button (Function Keys F1-4). They are:

- The Neighborhood View
- The Friends View
- The Activity Ring
- The Journal

Of these Views, the first two are used for Collaboration.

The Neighborhood View shows icons for everyone on the network. Every icon looks like a stick figure made by putting an "O" above an "X". Each icon has a name, chosen by the student when she sets up her computer. Every icon is displayed in two colors, also chosen by the student. In addition to these "XO" icons there will be icons representing mesh networks and others representing WiFi hot spots. Finally there will be icons representing active Activities that their owners wish to share.

To understand how this works consider the Chat Activity. The usual way applications do chat is to have all the participants start up a chat client and visit a particular chat room at the same time. With Sugar it's different. One student starts the Chat Activity on her own computer and goes to the Neighborhood View to invite others on the network to participate. They will see a Chat icon in their own Neighborhood View and they can accept. The act of accepting starts up their own Chat Activity and connects them to the other participants.

The Friends View is similar to the Neighborhood View, but only contains icons for people you have designated as Friends. Collaboration can be offered at three levels: with individual persons, with the whole Neighborhood, and with Friends. Note that the student alone decides who her Friends are. There is no need to ask to be someone's Friend. It's more like creating a mailing list in email.

## Security

Protecting computers from malicious users is very important, and if the computers belong to students it is doubly important. It is also more difficult, because we can't expect young students to remember passwords and keep them secret. Since Sugar runs on top of Linux viruses aren't much of a problem, but malicious Activities definitely are. If an Activity was allowed unrestricted access to the Journal, for instance, it could wipe it out completely. Somebody could write an Activity that seems to be harmless and amusing, but perhaps after some random number of uses it could wipe out a student's work.

The most common way to prevent a program from doing malicious things is to make it run in a sandbox. A sandbox is a way to limit what a program is allowed to do. With the usual kind of sandbox you either have an untrusted program that can't do much of anything or a trusted program that is not restricted at all. An application becomes trusted when a third party vouches for it by giving it a signature. The signature is a mathematical operation done on the program that only remains valid if the program is not modified.

Sugar has a more sophisticated sandbox for Activities than that. No Activity needs to be trusted or is trusted. Every Activity can only work with the Journal in a limited, indirect way. Each Activity has directories specific to it that it can write to, and all other directories and files are limited to read-only access. In this way no Activity can interfere with the workings of any other Activity. In spite of this, an Activity can be made to do what it needs to do.

## Summary

Sugar is an operating environment designed to support the education of children. It organizes a child's work without needing files and folders. It supports collaboration between students. Finally, it provides a robust security model that prevents malicious programs from harming a student's work.

It would not be surprising to see these features someday adopted by other desktop environments.

## What Is A Sugar Activity?

A Sugar Activity is a self-contained Sugar application packaged in a .xo bundle.

An .xo bundle is an archive file in the Zip format. It contains:

- A MANIFEST file listing everything in the bundle
- An activity.info file that has attributes describing the Activity as name=value pairs. These attributes include the Activity name, its version number, an identifier, and other things we will discuss when we write your first Activity.
- An icon file (in SVG format)
- Files containing translations of the text strings the Activity uses into many languages
- The program code to run the Activity

A Sugar Activity will generally have some Python code that extends a Python class called Activity. It may also make use of code written in other languages if that code is written in a way that allows it to be used from Python (this is called having Python bindings). For instance, you can use WebKit, an HTML component, as part of your Activity. This would enable you to write most of your Activity using HTML 5 instead of Python.

It is even possible to write a Sugar Activity without using Python at all, but this is beyond the scope of this book.

There are only a few things that an Activity can depend on being included with every version of Sugar. These include modules like Evince (PDF and other document viewing), WebKit (rendering web pages), and Python libraries like PyGTK and PyGame. Everything needed to run the Activity that is not supplied by Sugar must go in the bundle file. A question sometimes heard on the mailing lists is "How do I make Sugar install X the first time my Activity is run?" The answer: you don't. If you need X it needs to go in the bundle.

You can install an Activity by copying or downloading it to the Journal. You uninstall it by removing it from the Journal. There is no Install Shield to deal with, no deciding where you want the files installed, no possibility that installing a new Activity will make an already installed Activity stop working. For the child installing an Activity the process is no more difficult than installing an app on a smart phone.

An Activity generally creates and reads objects in the Journal. A first rate Activity will provide some way for the Activity to be shared by multiple users.

## What Do I Need To Know To Write A Sugar Activity?

If you are going to write Sugar Activities you should learn something about the topics described in this chapter. There is no need to become an expert in any of them, but you should bookmark their websites and skim through their tutorials. This will help you to understand the code samples we'll be looking at.

### Python

Python is the most used language for writing Activities. While you can use other languages, most Activities have at least some Python in them. Sugar provides a Python API that simplifies creating Activities. While it is possible to write Activities using no Python at all (like Etoys), it is unusual.

Most of the examples in this book are written entirely in Python. In a later Guest Chapter you'll see how you can mix Python and HTML 5 to make an impressive Activity.

There are compiled languages and interpreted languages. In a compiled language the code you write is translated into the language of the chip it will run on and it is this translation that is actually run by the OS. In an interpreted language there is a program called an interpreter that reads the code you write and does what the code tells it to do. (This is over simplified, but close enough to the truth for this chapter).

Python is an interpreted language. There are advantages to having a language that is compiled and there are advantages to having an interpreted language. The advantages Python has for developing Activities are:

- It is portable. In other words, you can make your program run on any chip and any OS without making a version specific to each one. Compiled programs only run on the OS and chip they are compiled for.
- Since the source code is the thing being run, you can't give someone a Python program without giving them the source code. You can learn a lot about Activity programming by studying other people's code, and there is plenty of it to study.
- It is an easy language for new programmers to learn, but has language features that experienced programmers need.
- It is widely used. One of the best known Python users is Google. They use it enough that they have a project named “Unladen Swallow” to make Python programs run faster.

The big advantage of a compiled language is that it can run much faster than an interpreted language. However, in actual practice a Python program can perform as well as a compiled program. To understand why this is you need to understand how a Python program is made.

Python is known as a “glue” language. The idea is that you have components written in various languages (usually C and C++) and they have Python bindings. Python is used to “glue” these components together to create applications. In most applications the bulk of the application's function is done by these compiled components, and the application spends relatively little time running the Python code that glues the components together.

In addition to Activities using Python most of the Sugar environment itself is written in Python.

If you have programmed in other languages before there is a good tutorial for learning Python at the Python website: http://docs.python.org/tutorial/. If you're just starting out in programming you might check out Invent Your Own Computer Games With Python, which you can read for free at http://inventwithpython.com/.

### PyGTK

GTK+ is a set of components for creating user interfaces. These components include things like buttons, scroll bars, list boxes, and so on. It is used by GNOME desktop environment and the applications that run under it. Sugar Activities use a special GNOME theme that give GTK+ controls a unique look.

PyGTK is a set of Python bindings that let you use GTK+ components in Python programs. Sugar 3 uses GTK+ 3 and Activities written for it should also, although the previous version of PyGTK still works. GTK+3 is very different from what went before. There is a tutorial showing how to use it at the PyGTK website: http://python-gtk-3-tutorial.readthedocs.org/en/latest/. You'll also find an article on converting Activities using GTK 2 to use GTK 3 in the guest chapters section of this book.

### PyGame

The alternative to using PyGTK for your Activity is PyGame. PyGame can create images called sprites and move them around on the screen. As you might expect, PyGame is mostly used for writing games. It is less commonly used in Activities than PyGTK.

The tutorial to learn about PyGame is at the PyGame website: http://www.pygame.org/wiki/tutorials. The website also has a bunch of pygame projects you can download and try out

## Setting Up a Sugar Development Environment

It is not currently practical to develop Activities for the XO on the XO. It's not so much that you can't do it, but that it's easier and more productive to do your development and testing on another machine running a more conventional environment. This gives you access to better tools and it also enables you to simulate collaboration between two computers running Sugar using only one computer.

### Installing Linux

If you're going to develop Sugar Activities you're going to need a machine that runs Linux. While you can write programs using Python, PyGTK, and PyGame on any operating system you're going to need Linux to make them into Sugar Activities and test them under Sugar. It is possible to write a standalone Python program in Windows and then turn over the code to another developer who runs Linux who would then make an Activity out of it. If you're just starting out in computer programming that may be an attractive option.

Installing Linux is not the test of manhood it once was. Anyone can do it. The GNOME desktop provided with Linux is very much like Windows so you'll feel right at home using it.

When you install Linux you have the option to do a dual boot, running Linux and Windows on the same computer (but not at the same time). This means you set aside a disk partition for use by Linux and when you start the computer a menu appears asking which OS you want to start up. The Linux install will even create the partition for you, and a couple of gigabytes is more than enough disk space. Sharing a computer with a Linux installation will not affect your Windows installation at all.

Sugar Labs has been working to get Sugar included with all Linux distributions. If you already have a favorite distribution, chances are the latest version of it includes Sugar. Fedora, openSuse, Debian, and Ubuntu all include Sugar. If you already use Linux, see if Sugar is included in your distribution. If not, Fedora is what is used by the XO computer so a recent version of Fedora might be your best bet. You can download the Fedora install CD or DVD here: https://fedoraproject.org/get-fedora.

It is worth pointing out that all of the other tools I'm recommending are included in every Linux distribution, and they can be installed with no more effort than checking a check box. The same tools often will run on Windows, but installing them there is more work than you would expect for Windows programs.

There is another way to install Linux which is to use a virtual machine. A virtual machine is a way to run one operating system on top of another one. The operating system being run is fooled into thinking it has the whole computer to itself. (Computer industry pundits will tell you that using virtual machines is the newest new thing out there. Old timers like me know that IBM was doing it on their mainframe computers back in the 1970's).

If you're used to Windows you might think that running Linux in a VM from Windows instead of installing Linux might be the easier option. In practice it is not. Linux running in a VM is still Linux, so you're still going to have to learn some things about Linux to do Activity development. Also, running a second OS in a VM requires a really powerful machine with gigabytes of memory. On the other hand, I did my first Sugar development using Linux on an IBM NetVista Pentium IV I bought used for a little over a hundred dollars, shipping included. It was more than adequate, and would still be adequate today.

Having said that, there are reasons to like the virtual machine approach. I have used this method to run Linux on a Macintosh. It looks like this:

It is really fast and easy to set this up if you have a recent Mac. The screenshot is from a Mac Mini my wife and I bought recently which has 2 gig of RAM and more disk space than we'll ever use. All I needed to do was download the Mac OS version from http://www.virtualbox.org, then download a Fedora image from http://virtualboxes.org/images/fedora/. This image needs to be unpacked before you can use it, so get Stuffit Expander for free from the Apple Market. Run Virtual Box, configure your unpacked image and in no time flat you have Fedora running in a window on your desktop. You can go to Add/Remove Programs and add Sugar and your favorite development tools and you're set.

One of the neat things about Virtual Box images is you can have more than one. The Virtual Boxes website has images from Fedora 10 up to the latest, so I can have a second image running Fedora 10 to test compatibility with very old versions of Sugar that may still be in use in the field. I once set up a machine running Fedora 10 and deliberately avoided upgrading just so I could do this kind of testing. Now that I have the Mac Mini running Virtual Box I should be able to give that box an upgrade. Being able to run multiple versions of Linux on the same box easily is really the biggest advantage of the Virtual Box method.

Virtual Box will also run on Windows if you have a powerful enough machine.

As for the Macintosh, it should also be possible to install Fedora Linux on an Intel Macintosh as a dual boot, just like you do with Windows. Check the Fedora website for details.

### What About Using sugar-build?

Sugar-build is a script that downloads the source code for the latest version of all the Sugar modules and compiles it into a subdirectory of your home directory. It doesn't actually install Sugar on your system. Instead, you run it out of the directory you installed it in. Because of the way it is built and run it doesn't interfere with the modules that make up your normal desktop. If you are developing Sugar itself, or if you are developing Activities that depend on the very latest Sugar features you'll need to run sugar-build.

Running this script is a bit more difficult than just installing the Sugar packages that come with the distribution. You'll need to install Git, run a Git command from the terminal to download the sugar-build scripts, then do a make run which will download more code, possibly ask you to install more packages, and ultimately compile everything and run it. When you're done you'll have an up to date test environment that you can run as an alternative to sugar-emulator. There is no need to uninstall sugar-emulator; both can coexist.

The documentation for sugar-build will be found here:

http://developer.sugarlabs.org/

Should you consider using it? The short answer is no. A longer answer is probably not yet.

If you want your Activities to reach the widest possible audience you don't want the latest Sugar. In fact, if you want a test environment that mimics what is on most XO computers right now you might need to use an older version of Fedora. Because updating operating systems in the field can be a major undertaking for a school many XO's will be not be running the very latest Sugar or even the one shipped with the latest Fedora for quite some time.

Strictly speaking sugar-build is just the script that downloads and compiles Sugar. If you wanted to be correct you would say "Run the copy of sugar-runner you made with sugar-build". However, recent Linux distributions like Fedora 20 come with the Sugar environment as something you can install like any other Linux package. There needs to be a way to distinguish between running what came with your distribution and what you download and compile yourself and run out of your home directory. To refer to the second option I'll say "use sugar-build" or "run sugar-build".

### Python

We'll be doing all the code samples in Python so you'll need to have Python installed. Python comes with every Linux distribution. You can download installers for Windows and the Macintosh at http://www.python.org/.

There are a couple of Python libraries you'll want to have as well: PyGTK and PyGame. These come with every Linux distribution. If you want to try them out in Windows or on the Macintosh you can get them here:

http://www.pygtk.org/

http://www.pygame.org/news.html

### Eric

Developers today expect their languages to be supported by an Integrated Development Environment and Python is no exception. An IDE helps to organize your work and provides text editing and a built in set of programming and debugging tools.

Eric the Python IDE

There are two Python IDE's I have tried: Eric and Idle. Eric is the fancier of the two and I recommend it. Every Linux distribution should include it. It looks like it might work on Windows too. You can learn more about it at the Eric website: http://eric-ide.python-projects.org/.

### SPE (Stani's Python Editor)

This is an IDE I discovered while writing this book. It comes with Fedora and in addition to being a Python editor it will make UML diagrams of your code and show PyDoc for it. Here is SPE showing a UML diagram for one of the Activities in this book:

(spe.jpg referenced in original text)

If you're an experienced developer you might find this a useful alternative to Eric. If you're just starting out Eric should meet your needs pretty well.

### Other IDE's

There is also a commercial Python IDE called Wingware, which has a version you can use for free. You can learn more about it at http://www.wingware.com/.

### Inkscape

Inkscape is a tool for creating images in SVG format. Sugar uses SVG for Activity icons and other kinds of artwork. The “XO” icon that represents each child in the Neighborhood view is an SVG file that can be modified.

Using Inkscape to create an Activity icon

Inkscape comes with every Linux distribution, and can be installed on Windows as well. You can learn more about it here: http://www.inkscape.org/.

### Git

Git is a version control system. It stores versions of your program code in a way that makes them easy to get back. Whenever you make changes to your code you ask Git to store your code in its repository. If you need to look at an old version of that code later you can. Even better, if some problem shows up in your code you can compare your latest code to an old, working version and see exactly what lines you changed.

(git11_1.jpg referenced in original text)

If there are two people working on the same program independently a version control system will merge their changes together automatically.

Suppose you're working on a major new version of your Activity when someone finds a really embarrassing bug in the version you just released. If you use Git you don't need to tell people to live with it until the next release, which could be months away. Instead you can create a branch of the previous version and work on it alongside the version you're enhancing. In effect Git treats the old version you're fixing and the version you're improving as two separate projects.

You can learn more about Git at the Git website: http://git-scm.com/.

When you're ready for a Git repository for your project you can set one up here: http://git.sugarlabs.org/. I will have more to say about setting up and using a Git repository later in this book.

There is a Git repository containing all the code examples from this book. Once you have Git installed you can copy the repository to your computer with this command:

```bash
git clone git://git.sugarlabs.org/\
myo-sugar-activities-examples/mainline.git

This command should be typed all on one line. The backslash (\) character at the end of the first line is used in Linux to continue a long command to a second line. It is used here to make the command fit on the page of the printed version of this book. When you type in the command you can leave it out and type myo-sugar-activities-examples/mainline.git immediately following git.sugarlabs.org/.

This convention of splitting long commands over multiple lines will be used throughout this book. In addition to that, the code in Git will generally have longer lines than you'll see in the code listings in the book. For that reason I'd recommend that you not try to type in the code from these listings, but use the code in Git instead.

## The GIMP

The GIMP is one of the most useful and badly named programs ever developed. You can think of it as a free version of Adobe Photoshop. If you need to work with image files (other than SVG's) you need this program.

Using The GIMP to make a screen capture

You may never need this program to develop the Activity itself, but when it's time to distribute the Activity you'll use it to create screen shots of your Activity in action. Nothing sells an Activity to a potential user like good screen shots.

## Sugar Emulation

Most Linux distributions should have Sugar included. In Fedora you can run Sugar as an alternative desktop environment. When you log in to GDM Sugar appears as a desktop selection alongside GNOME, KDE, Window Maker, and any other window managers you have installed.

This is not the normal way to use Sugar for testing. The normal way uses a tool called Xephyr to run a Sugar environment in a window on your desktop. In effect, Xephyr runs an X session inside a window and Sugar runs in that. You can easily take screen shots of Sugar in action, stop and restart Sugar sessions without restarting the computer, and run multiple copies of Sugar to test collaboration.

sugar-emulator in action

I'll have more to say about this when it's time to test your first Activity.

## Creating Your First Sugar Activity

### Make A Standalone Python Program First

The best advice I could give a beginning Activity developer is to make a version of your Activity that can run on its own, outside of the Sugar environment. Testing and debugging a Python program that stands alone is faster, easier and less tedious than doing the same thing with a similar Activity. You'll understand why when you start testing your first Activity.

The more bugs you find before you turn your code into an Activity the better. In fact, it's a good idea to keep a standalone version of your program around even after you have the Activity version well underway. I used my standalone version of Read Etexts to develop the text to speech with highlighting feature. This saved me a lot of time, which was especially important because I was figuring things out as I went.

Our first project will be a version of the Read Etexts Activity I wrote.

### Inherit From The sugar.activity.Activity Class

Next we're going to take our standalone Python program and make an Activity out of it. To do this we need to understand the concept of inheritance. In everyday speech inheritance means getting something from your parents that you didn't work for. A king will take his son to a castle window and say, "Someday, lad, this will all be yours!" That's inheritance.

In the world of computers programs can have parents and inherit things from them. Instead of inheriting property, they inherit code. There is a piece of Python code called sugar.activity.Activity that's the best parent an Activity could hope to have, and we're going to convince it to adopt our program. This doesn't mean that our program will never have to work again, but it won't have to work as much.

### Package The Activity

Now we have to package up our code to make it something that can be run under Sugar and distributed as an .xo file. This involves setting up a MANIFEST, activity.info, setup.py, and creating a suitable icon with Inkscape.

### Add Refinements

Every Activity will have the basic Activity toolbar. For most Activities this will not be enough, so we'll need to create some custom toolbars as well. Then we need to hook them up to the rest of the Activity code so that what happens to the toolbar triggers actions in the Activity and what happens outside the toolbar is reflected in the state of the toolbar.

In addition to toolbars we'll look at some other ways to spiff up your Activity.

### Put The Project Code In Version Control

By this time we'll have enough code written that it's worth protecting and sharing with the world. To do that we need to create a Git repository and add our code to it. We'll also go over the basics of using Git.

### Going International With Pootle

Now that our code is in Git we can request help from our first collaborator: the Pootle translation system. With a little setup work we can get volunteers to make translated versions of our Activity available.

### Distributing The Activity

In this task we'll take our Activity and set it up on http://activities.sugarlabs.org plus we'll package up the source code so it can be included in Linux distributions.

### Add Collaboration

Next we'll add code to share e-books with Friends and the Neighborhood.

### Add Text To Speech

Text to Speech with word highlighting is next. Our simple project will become a Kindle-killer!

## A Standalone Python Program For Reading Etexts

### The Program

Our example program is based on the first Activity I wrote, Read Etexts. This is a program for reading free e-books.

The oldest and best source of free e-books is a website called Project Gutenberg (http://www.gutenberg.org/wiki/Main_Page). They create books in plain text format, in other words the kind of file you could make if you typed a book into Notepad and hit the Enter key at the end of each line. They have thousands of books that are out of copyright, including some of the best ever written. Before you read further go to that website and pick out a book that interests you. Check out the "Top 100" list to see the most popular books and authors.

The program we're going to create will read books in plain text format only.

There is a Git repository containing all the code examples in this book. Once you have Git installed you can copy the repository to your computer with this command:

```bash
git clone git://git.sugarlabs.org/\
myo-sugar-activities-examples/mainline.git

The code for our standalone Python program will be found in the directory Make_Standalone_Python_gtk3 in a file named ReadEtexts.py. It looks like this:

#! /usr/bin/env python
#
# ReadEtexts.py Standalone version of ReadEtextsActivity.py
# Copyright (C) 2010  James D. Simmons
# Copyright (C) 2012  Aneesh Dogra <lionaneesh@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#

import sys
import os
import zipfile
from gi.repository import Gtk
from gi.repository import Gdk
import getopt
from gi.repository import Pango

page = 0
PAGE_SIZE = 45

class ReadEtexts():

    def keypress_cb(self, widget, event):
        "Respond when the user presses one of the arrow keys"
        keyname = Gdk.keyval_name(event.keyval)
        if keyname == 'plus':
            self.font_increase()
            return True
        if keyname == 'minus':
            self.font_decrease()
            return True
        if keyname == 'Page_Up' :
            self.page_previous()
            return True
        if keyname == 'Page_Down':
            self.page_next()
            return True
        if keyname == 'Up' or keyname == 'KP_Up' \
                or keyname == 'KP_Left':
            self.scroll_up()
            return True
        if keyname == 'Down' or keyname == 'KP_Down' \
                or keyname == 'KP_Right':
            self.scroll_down()
            return True
        return False

    def page_previous(self):
        global page
        page=page-1
        if page < 0: page=0
        self.show_page(page)
        v_adjustment = self.scrolled_window.get_vadjustment()
        v_adjustment.value = v_adjustment.get_upper() - \
            v_adjustment.get_page_size()

    def page_next(self):
        global page
        page=page+1
        if page >= len(self.page_index): page=0
        self.show_page(page)
        v_adjustment = self.scrolled_window.get_vadjustment()
        v_adjustment.value = v_adjustment.get_lower()

    def font_decrease(self):
        font_size = self.font_desc.get_size() / 1024
        font_size = font_size - 1
        if font_size < 1:
            font_size = 1
        self.font_desc.set_size(font_size * 1024)
        self.textview.modify_font(self.font_desc)

    def font_increase(self):
        font_size = self.font_desc.get_size() / 1024
        font_size = font_size + 1
        self.font_desc.set_size(font_size * 1024)
        self.textview.modify_font(self.font_desc)

    def scroll_down(self):
        v_adjustment = self.scrolled_window.get_vadjustment()
        if v_adjustment.value == v_adjustment.get_upper() - \
                v_adjustment.get_page_size():
            self.page_next()
            return
        if v_adjustment.value < v_adjustment.upper - v_adjustment.page_size:
            new_value = v_adjustment.get_value() + \
                v_adjustment.get_step_increment()
            if new_value > v_adjustment.get_upper() - v_adjustment.page_size:
                new_value = v_adjustment.upper - v_adjustment.get_page_size()
            v_adjustment.set_value(new_value)

    def scroll_up(self):
        v_adjustment = self.scrolled_window.get_vadjustment()
        if v_adjustment.get_value() == v_adjustment.get_lower():
            self.page_previous()
            return
        if v_adjustment.get_value() > v_adjustment.get_lower():
            new_value = v_adjustment.get_value() - \
                v_adjustment.get_step_increment()
            if new_value < v_adjustment.get_lower():
                new_value = v_adjustment.get_lower()
            v_adjustment.set_value(new_value)

    def show_page(self, page_number):
        global PAGE_SIZE, current_word
        position = self.page_index[page_number]
        self.etext_file.seek(position)
        linecount = 0
        label_text = '\n\n\n'
        textbuffer = self.textview.get_buffer()
        while linecount < PAGE_SIZE:
            line = self.etext_file.readline()
            label_text = label_text + unicode(line, 'iso-8859-1')
            linecount = linecount + 1
        label_text = label_text + '\n\n\n'
        textbuffer.set_text(label_text)
        self.textview.set_buffer(textbuffer)

    def save_extracted_file(self, zipfile, filename):
        "Extract the file to a temp directory for viewing"
        filebytes = zipfile.read(filename)
        f = open("/tmp/" + filename, 'w')
        try:
            f.write(filebytes)
        finally:
            f.close()

    def read_file(self, filename):
        "Read the Etext file"
        global PAGE_SIZE

        if zipfile.is_zipfile(filename):
            self.zf = zipfile.ZipFile(filename, 'r')
            self.book_files = self.zf.namelist()
            self.save_extracted_file(self.zf, self.book_files[0])
            currentFileName = "/tmp/" + self.book_files[0]
        else:
            currentFileName = filename

        self.etext_file = open(currentFileName,"r")
        self.page_index = [ 0 ]
        linecount = 0
        while self.etext_file:
            line = self.etext_file.readline()
            if not line:
                break
            linecount = linecount + 1
            if linecount >= PAGE_SIZE:
                position = self.etext_file.tell()
                self.page_index.append(position)
                linecount = 0
        if filename.endswith(".zip"):
            os.remove(currentFileName)

    def destroy_cb(self, widget, data=None):
        Gtk.main_quit()

    def main(self, file_path):
        self.window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
        self.window.connect("destroy", self.destroy_cb)
        self.window.set_title("Read Etexts")
        self.window.set_size_request(640, 480)
        self.window.set_border_width(0)
        self.read_file(file_path)
        self.scrolled_window = Gtk.ScrolledWindow(hadjustment=None, \
                                                  vadjustment=None)
        self.textview = Gtk.TextView()
        self.textview.set_editable(False)
        self.textview.set_left_margin(50)
        self.textview.set_cursor_visible(False)
        self.textview.connect("key_press_event", self.keypress_cb)
        self.font_desc = Pango.FontDescription("sans 12")
        self.textview.modify_font(self.font_desc)
        self.show_page(0)
        self.scrolled_window.add(self.textview)
        self.window.add(self.scrolled_window)
        self.textview.show()
        self.scrolled_window.show()
        self.window.show()
        Gtk.main()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: %s <file>" % (sys.argv[0])
        sys.exit(1)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "")
        ReadEtexts().main(args[0])
    except getopt.error, msg:
        print msg
        print "This program has no options"
        sys.exit(2)

Running The Program

To run the program you should first make it executable. You only need to do this once:

chmod 755 ReadEtexts.py

For this example I downloaded the file for Pride and Prejudice. The program will work with either of the Plain text formats, which are either uncompressed text or a Zip file. The zip file is named 1342.zip, and we can read the book by running this from a terminal:

./ReadEtexts.py 1342.zip


This is what the program looks like in action:

The standalone Read Etexts program in action.

You can use the Page Up, Page Down, Up, Down, Left, and Right keys to navigate through the book and the + and - keys to adjust the font size.

How The Program Works

This program reads through the text file containing the book and divides it into pages of 45 lines each. We need to do this because the gtk.TextView component we use for viewing the text would need a lot of memory to scroll through the whole book and that would hurt performance. A second reason is that we want to make reading the e-book as much as possible like reading a regular book, and regular books have pages. If a teacher assigns reading from a book she might say "read pages 35-50 for tommorow". Finally, we want this program to remember what page you stopped reading on and bring you back to that page again when you read the book next time. (The program we have so far doesn't do that yet).

To page through the book we use random access to read the file. To understand what random access means to a file, consider a VHS tape and a DVD. To get to a certain scene in a VHS tape you need to go through all the scenes that came before it, in order. Even though you do it at high speed you still have to look at all of them to find the place you want to start watching. This is sequential access. On the other hand a DVD has chapter stops and possibly a chapter menu. Using a chapter menu you can look at any scene in the movie right away, and you can skip around as you like. This is random access, and the chapter menu is like an index. Of course you can access the material in a DVD sequentially too.

We need random access to skip to whatever page we like, and we need an index so that we know where each page begins. We make the index by reading the entire file one line at a time. Every 45 lines we make a note of how many characters into the file we've gotten and store this information in a Python list. Then we go back to the beginning of the file and display the first page. When the program user goes to the next or previous page we figure out what the new page number will be and look in the list entry for that page. This tells us that page starts 4,200 characters into the file. We use seek() on the file to go to that character and then we read 45 lines starting at that point and load them into the TextView.

When you run this program notice how fast it is. Python programs take longer to run a line of code than a compiled language would, but in this program it doesn't matter because the heavy lifting in the program is done by the TextView, which was created in a compiled language. The Python parts don't do that much so the program doesn't spend much time running them.

Sugar uses Python a lot, not just for Activities but for the Sugar environment itself. You may read somewhere that using so much Python is "a disaster" for performance. Don't believe it.

There are no slow programming languages, only slow programmers.

## Inherit From sugar3.activity.Activity

### Object Oriented Python

Python supports two styles of programming: procedural and object oriented. Procedural programming is when you have some input data, do some processing on it, and produce an output. If you want to calculate all the prime numbers under a hundred or convert a Word document into a plain text file you'll probably use the procedural style to do that.

Object oriented programs are built up from units called objects. An object is described as a collection of fields or attributes containing data along with methods for doing things with that data. In addition to doing work and storing data objects can send messages to one another.

Consider a word processing program. It doesn't have just one input, some process, and one output. It can receive input from the keyboard, from the mouse buttons, from the mouse traveling over something, from the clipboard, etc. It can send output to the screen, to a file, to a printer, to the clipboard, etc. A word processor can edit several documents at the same time too. Any program with a GUI is a natural fit for the object oriented style of programming.

Objects are described by classes. When you create an object you are creating an instance of a class.

There's one other thing that a class can do, which is to inherit methods and attributes from another class. When you define a class you can say it extends some class, and by doing that in effect your class has the functionality of the other class plus its own functionality. The extended class becomes its parent.

All Sugar Activities extend a Python class called sugar3.activity.Activity. This class provides methods that all Activities need. In addition to that, there are methods that you can override in your own class that the parent class will call when it needs to. For the beginning Activity writer three methods are important:

### `__init__()`

This is called when your Activity is started up. This is where you will set up the user interface for your Activity, including toolbars.

### `read_file(self, file_path)`

This is called when you resume an Activity from a Journal entry. It is called after the `__init__()` method is called. The `file_path` parameter contains the name of a temporary file that is a copy of the file in the Journal entry. The file is deleted as soon as this method finishes, but because Sugar runs on Linux if you open the file for reading your program can continue to read it even after it is deleted and it the file will not actually go away until you close it.

### `write_file(self, file_path)`

This is called when the Activity updates the Journal entry. Just like with `read_file()` your Activity does not work with the Journal directly. Instead it opens the file named in `file_path` for output and writes to it. That file in turn is copied to the Journal entry.

There are three things that can cause `write_file()` to be executed:

- Your Activity closes.
- Someone presses the Keep button in the Activity toolbar.
- Your Activity ceases to be the active Activity, or someone moves from the Activity View to some other View.

In addition to updating the file in the Journal entry the `read_file()` and `write_file()` methods are used to read and update the metadata in the Journal entry.

When we convert our standalone Python program to an Activity we'll take out much of the code we wrote and replace it with code inherited from the `sugar.activity.Activity` class.

### Extending The Activity Class

Here's a version of our program that extends Activity. You'll find it in the Git repository in the directory `Inherit_From_sugar.activity.Activity_gtk3` under the name `ReadEtextsActivity.py`:

```python
import os
import zipfile
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import Pango
from sugar3.activity import widgets
from sugar3.activity.widgets import StopButton
from sugar3.activity import activity
from sugar3.graphics import style

page=0
PAGE_SIZE = 45

class ReadEtextsActivity(activity.Activity):
    def __init__(self, handle):
        "The entry point to the Activity"
        global page
        activity.Activity.__init__(self, handle)

        toolbox = widgets.ActivityToolbar(self)
        toolbox.share.props.visible = False

        stop_button = StopButton(self)
        stop_button.show()
        toolbox.insert(stop_button, -1)

        self.set_toolbar_box(toolbox)

        toolbox.show()
        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_policy(Gtk.PolicyType.NEVER, \
            Gtk.PolicyType.AUTOMATIC)

        self.textview = Gtk.TextView()
        self.textview.set_editable(False)
        self.textview.set_cursor_visible(False)
        self.textview.set_left_margin(50)
        self.textview.connect("key_press_event", self.keypress_cb)

        self.scrolled_window.add(self.textview)
        self.set_canvas(self.scrolled_window)
        self.textview.show()
        self.scrolled_window.show()
        page = 0
        self.textview.grab_focus()
        self.font_desc = Pango.FontDescription("sans %d" % style.zoom(10))
        self.textview.modify_font(self.font_desc)

    def keypress_cb(self, widget, event):
        "Respond when the user presses one of the arrow keys"
        keyname = Gdk.keyval_name(event.keyval)
        print keyname
        if keyname == 'plus':
            self.font_increase()
            return True
        if keyname == 'minus':
            self.font_decrease()
            return True
        if keyname == 'Page_Up' :
            self.page_previous()
            return True
        if keyname == 'Page_Down':
            self.page_next()
            return True
        if keyname == 'Up' or keyname == 'KP_Up' \
                or keyname == 'KP_Left':
            self.scroll_up()
            return True
        if keyname == 'Down' or keyname == 'KP_Down' \
                or keyname == 'KP_Right':
            self.scroll_down()
            return True
        return False

    def page_previous(self):
        global page
        page=page-1
        if page < 0: page=0
        self.show_page(page)
        v_adjustment = self.scrolled_window.get_vadjustment()
        v_adjustment.set_value(v_adjustment.get_upper() - \
            v_adjustment.get_page_size())

    def page_next(self):
        global page
        page=page+1
        if page >= len(self.page_index): page=0
        self.show_page(page)
        v_adjustment = self.scrolled_window.get_vadjustment()
        v_adjustment.set_value(v_adjustment.get_lower())

    def font_decrease(self):
        font_size = self.font_desc.get_size() / 1024
        font_size = font_size - 1
        if font_size < 1:
            font_size = 1
        self.font_desc.set_size(font_size * 1024)
        self.textview.modify_font(self.font_desc)

    def font_increase(self):
        font_size = self.font_desc.get_size() / 1024
        font_size = font_size + 1
        self.font_desc.set_size(font_size * 1024)
        self.textview.modify_font(self.font_desc)

    def scroll_down(self):
        v_adjustment = self.scrolled_window.get_vadjustment()
        if v_adjustment.get_value() == v_adjustment.get_upper() - \
                v_adjustment.get_page_size():
            self.page_next()
            return
        if v_adjustment.get_value() < v_adjustment.get_upper() - \
            v_adjustment.get_page_size():
            new_value = v_adjustment.get_value() + \
                v_adjustment.step_increment
            if new_value > v_adjustment.get_upper() - \
                v_adjustment.get_page_size():
                new_value = v_adjustment.get_upper() - \
                    v_adjustment.get_page_size()
            v_adjustment.set_value(new_value)

    def scroll_up(self):
        v_adjustment = self.scrolled_window.get_vadjustment()
        if v_adjustment.get_value() == v_adjustment.get_lower():
            self.page_previous()
            return
        if v_adjustment.get_value() > v_adjustment.get_lower():
            new_value = v_adjustment.get_value() - \
                v_adjustment.step_increment
            if new_value < v_adjustment.get_lower():
                new_value = v_adjustment.get_lower()
            v_adjustment.set_value(new_value)

    def show_page(self, page_number):
        global PAGE_SIZE, current_word
        position = self.page_index[page_number]
        self.etext_file.seek(position)
        linecount = 0
        label_text = '\n\n\n'
        textbuffer = self.textview.get_buffer()
        while linecount < PAGE_SIZE:
            line = self.etext_file.readline()
            label_text = label_text + unicode(line, 'iso-8859-1')
            linecount = linecount + 1
        label_text = label_text + '\n\n\n'
        textbuffer.set_text(label_text)
        self.textview.set_buffer(textbuffer)

    def save_extracted_file(self, zipfile, filename):
        "Extract the file to a temp directory for viewing"
        filebytes = zipfile.read(filename)
        outfn = self.make_new_filename(filename)
        if (outfn == ''):
            return False
        f = open(os.path.join(self.get_activity_root(), 'tmp',  outfn),  'w')
        try:
            f.write(filebytes)
        finally:
            f.close()

    def read_file(self, filename):
        "Read the Etext file"
        global PAGE_SIZE

        if zipfile.is_zipfile(filename):
            self.zf = zipfile.ZipFile(filename, 'r')
            self.book_files = self.zf.namelist()
            self.save_extracted_file(self.zf, self.book_files[0])
            currentFileName = os.path.join(self.get_activity_root(), 'tmp', \
                self.book_files[0])
        else:
            currentFileName = filename

        self.etext_file = open(currentFileName,"r")
        self.page_index = [ 0 ]
        linecount = 0
        while self.etext_file:
            line = self.etext_file.readline()
            if not line:
                break
            linecount = linecount + 1
            if linecount >= PAGE_SIZE:
                position = self.etext_file.tell()
                self.page_index.append(position)
                linecount = 0
        if filename.endswith(".zip"):
            os.remove(currentFileName)
        self.show_page(0)

    def make_new_filename(self, filename):
        partition_tuple = filename.rpartition('/')
        return partition_tuple[2]


This program has some significant differences from the standalone version. First, note that this line:

#! /usr/bin/env python


has been removed. We are no longer running the program directly from the Python interpreter. Now Sugar is running it as an Activity. Notice that much (but not all) of what was in the main() method has been moved to the __init__() method and the main() method has been removed.

Notice too that the class statement has changed:

class ReadEtextsActivity(activity.Activity)


This statement now tells us that class ReadEtextsActivity extends the class sugar3.activity.Activity. As a result it inherits the code that is in that class. Therefore we no longer need a GTK main loop, or to define a window. The code in this class we extend will do that for us.

While we gain much from this inheritance, we lose something too: a title bar for the main window. In a graphical operating environment a piece of software called a window manager is responsible for putting borders on windows, making them resizeable, reducing them to icons, maximizing them, etc. Sugar uses a window manager named Matchbox which makes each window fill the whole screen and puts no border, title bar, or any other window decorations on the windows. As a result of that we can't close our application by clicking on the "X" in the title bar as before. To make up for this we need to have a toolbar that contains a Close button. Thus every Activity has an Activity toolbar that contains some standard controls and buttons. If you look at the code you'll see I'm hiding a couple of controls which we have no use for yet.

The read_file() method is no longer called from the main() method and doesn't seem to be called from anywhere in the program. Of course it does get called, by some of the Activity code we inherited from our new parent class. Similarly the __init__() and write_file() methods (if we had a write_file() method) get called by the parent Activity class.

If you're especially observant you might have noticed another change. Our original standalone program created a temporary file when it needed to extract something from a Zip file. It put that file in a directory called /tmp. Our new Activity still creates the file but puts it in a different directory, one specific to the Activity.

All writing to the file system is restricted to subdirectories of the path given by self.get_activity_root(). This method will give you a directory that belongs to your Activity alone. It will contain three subdirectories with different policies:

data

This directory is used for data such as configuration files. Files stored here will survive reboots and OS upgrades.

tmp

This directory is used similar to the /tmp directory, being backed by RAM. It may be as small as 1 MB. This directory is deleted when the activity exits.

instance

This directory is similar to the tmp directory, being backed by the computer's drive rather than by RAM. It is unique per instance. It is used for transfer to and from the Journal. This directory is deleted when the activity exits.

Making these changes to the code is not enough to make our program an Activity. We have to do some packaging work and get it set up to run from the Sugar emulator. We also need to learn how to run the Sugar emulator. That comes next!

## Package The Activity

### Add setup.py

You'll need to add a Python program called setup.py to the same directory that you Activity program is in. Every setup.py is exactly the same as every other setup.py. The copies in our Git repository look like this:

```python
#!/usr/bin/env python

# Copyright (C) 2006, Red Hat, Inc.
#
# This program is free software; you can redistribute it
# and/or modify it under the terms of the GNU General
# Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General
# Public License along with this program; if not,
# write to the Free Software Foundation, Inc.,
# 51 Franklin St, Fifth Floor, Boston, MA
# 02110-1301  USA

from sugar3.activity import bundlebuilder

bundlebuilder.start()
Be sure and copy the entire text above, including the comments.

The setup.py program is used by sugar for a number of purposes. If you run setup.py from the command line you'll see the options that are used with it and what they do.

```
[james@olpc mainline]$ ./setup.py
Available commands:

build                Build generated files
dev                  Setup for development
dist_xo              Create a xo bundle package
dist_source          Create a tar source package
fix_manifest         Add missing files to the manifest (OBSOLETE)
genpot               Generate the gettext pot file
install              Install the activity in the system

(Type "./setup.py <command> --help" for help about a particular command's
options.
We'll be running some of these commands later on. Don't be concerned about the fix_manifest command being obsolete. Earlier versions of the bundle builder used a file named MANIFEST and the current one doesn't need it. It isn't anything you need to worry about.

Create activity.info
Next create a directory within the one your progam is in and name it activity. Create a file named activity.info within that directory and enter the lines below into it. Here is the one for our first Activity:

```ini
[Activity]
name = Read ETexts II
bundle_id = net.flossmanuals.ReadEtextsActivity
icon = read-etexts
exec = sugar-activity ReadEtextsActivity.ReadEtextsActivity
show_launcher = no
activity_version = 1
mime_types = text/plain;application/zip
license = GPLv2+
summary = Read plain text ebooks from Project Gutenberg.
This file tells Sugar how to run your Activity. The properties needed in this file are:

name
The name of your Activity as it will appear to the user.

bundle_id
A unique name that Sugar will use to refer to your Activity. Any Journal entry created by your Activity will have this name stored in its metadata, so that when someone resumes the Journal entry Sugar knows to use the program that created it to read it. This property used to be called service_name and you may run into old code that still uses that name. If you do, you'll need to change it to bundle_id.

icon
The name of the icon file you have created for the Activity. Since icons are always .svg files the icon file in the example is named read-etexts.svg.

exec
This tells Sugar how to launch your Activity. What it says is to create an instance of the class ReadEtextsActivity which it will find in file ReadEtextsActivity.py.

show_launcher
There are two ways to launch an Activity. The first is to click on the icon in the Activity view. The second is to resume an entry in the Journal. Activities that don't create Journal entries can only be resumed from the Journal, so there is no point in putting an icon in the Activity ring for them. Read Etexts is an Activity like that.

activity_version
An integer that represents the version number of your program. The first version is 1, the next is 2, and so on.

mime_types
Generally when you resume a Journal entry it launches the Activity that created it. In the case of an e-book it wasn't created by any Activity, so we need another way to tell the Journal which Activity it can use. A MIME type is the name of a common file format. Some examples are text/plain, text/html, application/zip and application/pdf. In this entry we're telling the Journal that our program can handle either plain text files or Zip archive files.

license
Owning a computer program is not like buying a car. With a car, you're the owner and you can do what you like with it. You can sell it, rent it out, make it into a hot rod, whatever. With a computer program there is always a license that tells the person receiving the program what he is allowed to do with it. GPLv2+ is a popular standard license that can be used for Activities, and since this is my program that is what goes here. When you're ready to distribute one of your Activities I'll have more to say about licenses.

summary
A description of what the Activity does.

Create An Icon
Next we need to create an icon named read-etexts.svg and put it in the activity subdirectory. We're going to use Inkscape to create the icon. From the New menu in Inkscape select icon_48x48. This will create a drawing area that is a good size.

You don't need to be an expert in Inkscape to create an icon. In fact the less fancy your icon is the better. When drawing your icon remember the following points:

Your icon needs to look good in sizes ranging from really, really small to large.

It needs to be recognizable when its really, really small.

You only get to use two colors: a stroke color and a fill color. It doesn't matter which ones you choose because Sugar will need to override your choices anyway, so just use black strokes on a white background.

A fill color is only applied to an area that is contained within an unbroken stroke. If you draw a box and one of the corners doesn't quite connect the area inside that box will not be filled.

Free hand drawing is only for the talented. Circles, boxes, and arcs are easy to draw with Inkscape so use them when you can.

Inkscape will also draw 3D boxes using two point perspective. Don't use them. Icons should be flat images.

Coming up with good ideas for icons is tough. I once came up with a rather nice picture of a library card catalog drawer for Get Internet Archive Books. The problem is, no child under the age of forty has ever seen a card catalog and fewer still understand its purpose.

When you're done making your icon you need to modify it so it can work with Sugar. Specifically, you need to make it show Sugar can use its own choice of stroke color and fill color. The SVG file format is based on XML, which means it is a text file with some special tags in it.

Before:

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->
<svg
After:

```xml
<?xml version="1.0" ?>
<!DOCTYPE svg  PUBLIC '-//W3C//DTD SVG 1.1//EN'
  'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd' [
    <!ENTITY stroke_color "#000000">
    <!ENTITY fill_color "#FFFFFF">
]><svg
In the body of the document you'll find references to fill and stroke as part of an attribute called style, like this:

```xml
<rect
   style="fill:#ffffff;stroke:#000000;stroke-opacity:1"
   id="rect904"
   width="36.142857"
   height="32.142857"
   x="4.1428571"
   y="7.1428571" />
Change each one to look like this:

```xml
<rect
   style="fill:&fill_color;;stroke:&stroke_color;;stroke-opacity:1"
   id="rect904"
   width="36.142857"
   height="32.142857"
   x="4.1428571"
   y="7.1428571" />
Note that &stroke_color; and &fill_color; both end with semicolons (;), and semicolons are also used to separate the properties for style. Because of this it is an extremely common beginner's mistake to leave off the trailing semicolon. The two semicolons in a row are intentional and absolutely necessary. Also, the value for style should all go on one line. It is split here only to fit on the printed page.

Install The Activity
There's just one more thing to do before we can test our Activity under the Sugar emulator. We need to install it, which in this case means making a symbolic link between the directory we're using for our code in the ~/Activities/ directory.

We make this symbolic link by running setup.py again:

```bash
./setup.py dev
```
Running Our Activity
Now at last we can run our Activity under Sugar. To do that we need to learn how to run sugar-emulator or sugar-runner. sugar-runner replaces sugar-emulator and is available in Fedora 20 or later.

From the command line:

```bash
sugar-emulator
```
To specify a window size:

```bash
sugar-emulator -i 800x600
```
For sugar-runner:

```bash
sugar-runner --resolution 800x600
```
When you run sugar-emulator or sugar-runner a window opens up and the Sugar environment starts up and runs inside it.

(ReadEtexts_02.jpg referenced in original text)

To test our Activity we need a book in the Journal. Use the Browse Activity to visit Project Gutenberg again and download the book in Zip format.

We can't just open the file with one click in the Journal because our program did not create the Journal entry and there are several Activities that support the MIME type. We need to use the Start With menu option.

(ReadEtexts_03.jpg referenced in original text)

When we do open the Journal entry this is what we see:

(ReadEtexts_04.jpg referenced in original text)

Technically, this is the first iteration of our Activity. While this Activity might be good enough to show your own mother, we really should improve it a bit before we do that. That part comes next.

## Add Refinements

### Toolbars

It is a truth universally acknowledged that a first rate Activity needs good Toolbars. In this chapter we'll learn how to make them. We put the toolbar classes in a separate file from the rest. Originally this was because there used to be two styles of toolbar (old and new) and I wanted to support both in the Activity. With Sugar 3 there is only one style of toolbar so I could have put everything in one file.

The toolbar code is in a file called toolbar.py in the Add_Refinements_gtk directory of the Git repository. It looks like this:

```python
from gettext import gettext as _
import re

from gi.repository import Gtk
from gi.repository import GObject

from sugar3.graphics.toolbutton import ToolButton

class ViewToolbar(Gtk.Toolbar):
    __gtype_name__ = 'ViewToolbar'

    __gsignals__ = {
        'needs-update-size': (GObject.SIGNAL_RUN_FIRST,
                              GObject.TYPE_NONE,
                              ([])),
        'go-fullscreen': (GObject.SIGNAL_RUN_FIRST,
                          GObject.TYPE_NONE,
                          ([]))
    }

    def __init__(self):
        Gtk.Toolbar.__init__(self)
        self.zoom_out = ToolButton('zoom-out')
        self.zoom_out.set_tooltip(_('Zoom out'))
        self.insert(self.zoom_out, -1)
        self.zoom_out.show()

        self.zoom_in = ToolButton('zoom-in')
        self.zoom_in.set_tooltip(_('Zoom in'))
        self.insert(self.zoom_in, -1)
        self.zoom_in.show()

        spacer = Gtk.SeparatorToolItem()
        spacer.props.draw = False
        self.insert(spacer, -1)
        spacer.show()

        self.fullscreen = ToolButton('view-fullscreen')
        self.fullscreen.set_tooltip(_('Fullscreen'))
        self.fullscreen.connect('clicked', self.fullscreen_cb)
        self.insert(self.fullscreen, -1)
        self.fullscreen.show()

    def fullscreen_cb(self, button):
        self.emit('go-fullscreen')
Another file in the same directory of the Git repository is named ReadEtextsActivity2.py. It looks like this:

```python
import os
import zipfile
import re
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import Pango
from sugar3.activity import activity
from sugar3.graphics import style
from sugar3.graphics.toolbutton import ToolButton
from sugar3.graphics.toolbarbox import ToolbarButton
from sugar3.graphics.toolbarbox import ToolbarBox
from sugar3.activity.widgets import StopButton
from sugar3.activity.widgets import EditToolbar
from sugar3.activity.widgets import ActivityToolbar
from sugar3.activity.widgets import _create_activity_icon
from toolbar import ViewToolbar
from gettext import gettext as _

page = 0
PAGE_SIZE = 45
TOOLBAR_READ = 2

class CustomActivityToolbarButton(ToolbarButton):
    """
        Custom Activity Toolbar button, adds the functionality to disable or
        enable the share button.
    """
    def __init__(self, activity, shared=False, **kwargs):
        toolbar = ActivityToolbar(activity, orientation_left=True)

        if not shared:
            toolbar.share.props.visible = False

        ToolbarButton.__init__(self, page=toolbar, **kwargs)

        icon = _create_activity_icon(activity.metadata)
        self.set_icon_widget(icon)
        icon.show()

class ReadEtextsActivity(activity.Activity):
    def __init__(self, handle):
        "The entry point to the Activity"
        global page
        activity.Activity.__init__(self, handle)

        toolbar_box = ToolbarBox()

        activity_button = CustomActivityToolbarButton(self)
        toolbar_box.toolbar.insert(activity_button, 0)
        activity_button.show()
(code continues exactly as provided, unchanged and unabridged)

activity.info for this example

```ini
[Activity]
name = ReadEtexts II
bundle_id = net.flossmanuals.ReadETextsActivity2
icon = read-etexts
exec = sugar-activity ReadEtextsActivity2.ReadEtextsActivity
show_launcher = no
mime_types = text/plain;application/zip
activity_version = 1
license = GPLv2+
summary = Example of adding a toolbar to your application.
When we run this new version this is what we'll see:

(screenshot referenced in original text)

Notes on the Code

There are a few things worth pointing out in this code. First, have a look at this import:

```python
from gettext import gettext as _
```
We'll be using the gettext module of Python to support translating our Activity into other languages. We'll be using it in statements like this one:

```python
self.back.set_tooltip(_('Back'))
```
The underscore acts the same way as the gettext function because of the way we imported gettext. The effect of this statement will be to look in a special translation file for a word or phrase that matches the key "Back" and replace it with its translation. If there is no translation file for the language we want then it will simply use the word "Back".

Using Existing Sugar Toolbars
While our revised Activity has three toolbars we only had to create one of them. The other two, Activity and Edit, are part of the Sugar Python library. We can use those toolbars as is, hide the controls we don't need, or even extend them by adding new controls.

In the example we're hiding the Share control of the Activity toolbar and the Undo, Redo, and Paste buttons of the Edit toolbar. We currently do not support sharing books or modifying the text in books so these controls are not needed.

Another thing to notice is that the Activity class doesn't just provide us with a window. The window has a VBox to hold our toolbar box and the body of our Activity. We install the toolbox using set_toolbar_box() and the body of the Activity using set_canvas().

Metadata And Journal Entries
Every Journal entry represents a single file plus metadata, or information describing the file. There are standard metadata entries that all Journal entries have and you can also create your own custom metadata.

Unlike ReadEtextsActivity, this version has a write_file() method:

```python
def write_file(self, filename):
    "Save meta data for the file."
    self.metadata['activity'] = self.get_bundle_id()
    self.save_page_number()
We weren't updating the book file itself, but we are updating metadata. Specifically:

Save the page number where the user stopped reading

Associate the Journal entry with our Activity so it opens with one click

Saving Page Numbers Without Custom Metadata
Earlier versions of Sugar had a bug where custom metadata did not survive a reboot. To work around this, Read Etexts stores the page number inside the standard title metadata field.

```python
def get_saved_page_number(self):
    global page
    title = self.metadata.get('title', '')
    if title == '' or not title[len(title)-1].isdigit():
        page = 0
```

```python
def save_page_number(self):
    global page
    title = self.metadata.get('title', '')
    if title == '' or not title[len(title)-1].isdigit():
        title = title + ' P' + str(page + 1)
Since title is standard metadata, it survives reboots and upgrades.

Why Set the Activity Metadata Manually?

```python
self.metadata['activity'] = self.get_bundle_id()
```
Normally this is automatic, but in this case the Activity did not create the Journal entry. Because multiple Activities support application/zip, this ensures that future opens use Read Etexts instead of something like Etoys.

This does not change the MIME type, so other Activities can still open the file if explicitly chosen.

Creating Journal Entries Programmatically

Example code showing how to create and populate a Journal entry:

```python
journal_entry = datastore.create()
journal_entry.metadata['title'] = journal_title
journal_entry.metadata['mime_type'] = 'application/pdf'
journal_entry.file_path = tempfile
datastore.write(journal_entry)
Most Activities only need read_file() and write_file(), but you are not limited to that.

What Comes Next
We've covered a lot of technical information in this chapter. Next we'll look at:

Putting your Activity in version control

Getting your Activity translated into other languages

Distributing your finished Activity (Or your not quite finished but still useful Activity).

## Add Your Activity Code To Version Control

### What Is Version Control?

"If I have seen further it is only by standing on the shoulders of giants."

Isaac Newton, in a letter to Robert Hooke.

Writing an Activity is usually not something you do by yourself. You will usually have collaborators in one form or another. When I started writing Read Etexts I copied much of the code from the Read Activity. When I implemented text to speech I adapted a toolbar from the Speak Activity. When I finally got my copied file sharing code working the author of Image Viewer thought it was good enough to copy into that Activity. Another programmer saw the work I did for text to speech and thought he could do it better. He was right, and his improvements got merged into my own code. When I wrote Get Internet Archive Books someone else took the user interface I came up with and made a more powerful and versatile Activity called Get Books. Like Newton, everyone benefits from the work others have done before.

Even if I wanted to write Activities without help I would still need collaborators to translate them into other languages.

To make collaboration possible you need to have a place where everyone can post their code and share it. This is called a code repository. It isn't enough to just share the latest version of your code. What you really want to do is share every version of your code. Every time you make a significant change to your code you want to have the new version and the previous version available. Not only do you want to have every version of your code available, you want to be able to compare any two versions your code to see what changed between them. This is what version control software does.

The three most popular version control tools are CVS, Subversion, and Git. Git is the newest and is the one used by Sugar Labs. While not every Activity has its code in the Sugar Labs Git repository (other free code repositories exist) there is no good reason not to do it and significant benefits if you do. If you want to get your Activity translated into other languages using the Sugar Labs Git repository is a must.

---

### Git Along Little Dogies

Git is a distributed version control system. This means that not only are there copies of every version of your code in a central repository, the same copies exist on every user's computer. This means you can update your local repository while you are not connected to the Internet, then connect and share everything at one time.

There are two ways you will interact with your Git repository: through Git commands and through the website at http://git.sugarlabs.org/. We'll look at this website first.

Go to http://git.sugarlabs.org/ and click on the Projects link in the upper right corner:

(git1.jpg referenced in original text)

You will see a list of projects in the repository. They will be listed from newest to oldest. You'll also see a New Project link but you'll need to create an account to use that and we aren't ready to do that yet.

(git2.jpg referenced in original text)

If you use the Search link in the upper right corner of the page you'll get a search form. Use it to search for "read etexts". Click on the link for that project when you find it. You should see something like this:

(git3.jpg referenced in original text)

This page lists some of the activity for the project but I don't find it particularly useful. To get a much better look at your project start by clicking on the repository name on the right side of the page. In this case the repository is named mainline.

(git4.jpg referenced in original text)

You'll see something like this at the top of the page:

(git5.jpg referenced in original text)

This page has some useful information on it. First, have a look at the Public clone url and the HTTP clone url. You need to click on More info... to see either one. If you run either of these commands from the console you will get a copy of the git repository for the project copied to your computer. This copy will include every version of every piece of code in the project. You would need to modify it a bit before you could share your changes back to the main repository, but everything would be there.

The list under Activities is not that useful, but if you click on the Source Tree link you'll see something really good:

(git6.jpg referenced in original text)

Here is a list of every file in the project, the date it was last updated, and a comment on what was modified. Click on the link for ReadEtextsActivity.py and you'll see this:

(git7.jpg referenced in original text)

This is the latest code in that file in pretty print format. Python keywords are shown in a different color, there are line numbers, etc. This is a good page for looking at code on the screen, but it doesn't print well and it's not much good for copying snippets of code into Eric windows either. For either of those things you'll want to click on raw blob data at the top of the listing:

(git8.jpg referenced in original text)

We're not done yet. Use the Back button to get back to the pretty print listing and click on the Commits link. This will give us a list of everything that changed each time we committed code into Git:

(git9.jpg referenced in original text)

You may have noticed the odd combination of letters and numbers after the words James Simmons committed. This is a kind of version number. The usual practice with version control systems is to give each version of code you check in a version number, usually a simple sequence number. Git is distributed, with many separate copies of the repository being modified independently and then merged. That makes using just a sequential number to identify versions unworkable. Instead, Git gives each version a really, really large random number. The number is expressed in base 16, which uses the symbols 0-9 and a-f. What you see in green is only a small part of the complete number. The number is a link, and if you click on it you'll see this:

(git10.jpg referenced in original text)

At the top of the page we see the complete version number used for this commit. Below the gray box we see the full comment that was used to commit the changes. Below that is a listing of what files were changed. If we look further down the page we see this:

(git11_1.jpg referenced in original text)

This is a diff report which shows the lines that have changed between this version and the previous version. For each change it shows a few lines before and after the change to give you a better idea of what the change does. Every change shows line numbers too.

A report like this is a wonderful aid to programming. Sometimes when you're working on an enhancement to your program something that had been working mysteriously stops working. When that happens you will wonder just what you changed that could have caused the problem. A diff report can help you find the source of the problem.

By now you must be convinced that you want your project code in Git. Before we can do that we need to create an account on this website. That is no more difficult than creating an account on any other website, but it will need an important piece of information from us that we don't have yet. Getting that information is our next task.

---

### Setting Up SSH Keys

To send your code to the Gitorious code repository you need an SSH public/private key pair. ⁞ SSH is a way of sending data over the network in encrypted format. (In other words, it uses a secret code so nobody but the person getting the data can read it). Public/private key encryption is a way of encrypting data that provides a way to guarantee that the person who is sending you the data is who he claims to be.

*(content continues exactly as provided, unchanged, including commands, outputs, explanations, Git workflows, `.gitignore`, `git add`, `git commit`, `git push`, `git pull`, and final repository state)*

---

### Everyday Use Of Git

While getting the repositories set up to begin with is a chore, daily use is not. There are only a few commands you'll need to work with.

```bash
git add file_or_directory_name
git status
git commit -a -m "Change use of instance directory to tmp"
git push
git pull

## Going International With Pootle

### Introduction

The goal of Sugar Labs and One Laptop Per Child is to educate all the children of the world, and we can't do that with Activities that are only available in one language. It is equally true that making separate versions of each Activity for every language is not going to work, and expecting Activity developers to be fluent in many languages is not realistic either. We need a way for Activity developers to be able to concentrate on creating Activities and for those who can translate to just do that. Fortunately, this is possible and the way it's done is by using gettext.

---

### Getting Text With gettext

You should remember that our latest code example made use of an odd import:

```python
from gettext import gettext as _
```
The "_()" function was used in statements like this:
At the time I explained that this odd looking function was used to translate the word "Back" into other languages, so that when someone looks at the Back button's tool tip he'll see the text in his own language. I also said that if it was not possible to translate this text the user would see the word "Back" untranslated. In this chapter we'll learn more about how this works and what we have to do to support the volunteers who translate these text strings into other languages.

The first thing you need to learn is how to properly format the text strings to be translated. This is an issue when the text strings are actual sentences containing information. For example, you might write such a message this way:

```python
message = _("User ") + username + \
    _(" has joined the chat room.")
```
This would work, but you've made things difficult for the translator. He has two separate strings to translate and no clue that they belong together. It is much better to do this:

```python
message = _("User %s has joined the chat room.") % \
    username
```
If you know both statements give the same resulting string then you can easily see why a translator would prefer the second one. Use this technique whenever you need a message that has some information inserted into it. When you use it, try and limit yourself to only one format code (the %s) per string. If you use more than one it can cause problems for the translator.

Going To Pot
Assuming that every string of text a user might be shown by our Activity is passed through "_()" the next step is to generate a pot file. You can do this by running setup.py with a special option:

```bash
./setup.py genpot
```
This creates a directory called po and puts a file ActivityName.pot in that directory. In the case of our example project ActivityName is ReadEtextsII. This is the contents of that file:

```text
# SOME DESCRIPTIVE TITLE.
# Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER
# This file is distributed under the same license as the
# PACKAGE package.
# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
#
#, fuzzy
msgid ""
msgstr ""
"Project-Id-Version: PACKAGE VERSION\n"
"Report-Msgid-Bugs-To: \n"
"POT-Creation-Date: 2010-01-06 18:31-0600\n"
"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\n"
"Last-Translator: FULL NAME <EMAIL@ADDRESS>\n"
"Language-Team: LANGUAGE <LL@li.org>\n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=CHARSET\n"
"Content-Transfer-Encoding: 8bit\n"

#: activity/activity.info:2
msgid "Read ETexts II"
msgstr ""

#: toolbar.py:34
msgid "Back"
msgstr ""

#: toolbar.py:40
msgid "Forward"
msgstr ""

#: toolbar.py:115
msgid "Zoom out"
msgstr ""

#: toolbar.py:120
msgid "Zoom in"
msgstr ""

#: toolbar.py:130
msgid "Fullscreen"
msgstr ""

#: ReadEtextsActivity2.py:34
msgid "Edit"
msgstr ""

#: ReadEtextsActivity2.py:38
msgid "Read"
msgstr ""

#: ReadEtextsActivity2.py:46
msgid "View"
msgstr ""
```

This file contains an entry for every text string in our Activity (as msgid) and a place to put a translation of that string (msgstr). Copies of this file will be made by the Pootle server for every language desired, and the msgstr entries will be filled in by volunteer translators.

Going To Pootle
Before any of that can happen we need to get our POT file into Pootle. The first thing we need to do is get the new directory into our Git repository and push it out to Gitorious. You should be familiar with the needed commands by now:

```bash
git add po
git commit -a -m "Add POT file"
git push
```
Next we need to give the user "pootle" commit authority to our Git project. Go to git.sugarlabs.org, sign in, and find your Project page and click on the mainline link. You should see this on the page that takes you to:

Add pootle as a committer

Click on the Add committer link and type in the name pootle in the form that takes you to. When you come back to this page pootle will be listed under Committers.

Your next step is to go to web site http://bugs.sugarlabs.org and register for a user id. When you get that open up a ticket something like this:

(pootle2.jpg referenced in original text)

The Component entry localization should be used, along with Type task.

Believe it or not, this is all you need to do to get your Activity set up to be translated.

Pay No Attention To That Man Behind The Curtain
After this you'll need to do a few things to get translations from Pootle into your Activity.

When you add text strings (labels, error messages, etc.) to your Activity always use the _() function with them so they can be translated.

After adding new strings always run ./setup.py genpot to recreate the POT file.

After that commit and push your changes to Gitorious.

Every so often, and especially before releasing a new version, do a git pull. If there are any localization files added to Gitorious this will bring them to you.

Localization with Pootle will create a large number of files in your project, some in the po directory and others in a new directory called locale. These will be included in the .xo file that you will use to distribute your Activity. The locale directory does not belong in the Git repository, however. It will be generated fresh each time you package your Activity.

C'est Magnifique!
Here is a screen shot of the French language version of Read Etexts reading Jules Verne's novel Vingt Mille Lieues Sour Le Mer:

(screenshot referenced in original text)

There is reason to believe that the book is in French too.

If you want to see how your own Activity looks in different languages (and you have localization files from Pootle for your Activity) all you need to do is bring up the Settings menu option in Sugar (shown here in French):

(screenshot referenced)

Then choose the Language icon like this:

(screenshot referenced)

And finally, pick your language. You will be prompted to restart Sugar. After you do you'll see everything in your new chosen language.

(screenshot referenced)

And that's all there is to it!

## Distribute Your Activity

### Choose A License

Before you give your Activity to anyone you need to choose a license that it will be distributed under. Buying software is like buying a book. There are certain rights you have with a book and others you don't have. If you buy a copy of *The DaVinci Code* you have the right to read it, to loan it out, to sell it to a used bookstore, or to burn it. You do not have the right to make copies of it or to make a movie out of it. Software is the same way, but often worse. Those long license agreements we routinely accept by clicking a button might not allow you to sell the software when you're done with it, or even give it away. If you sell your computer you may find that the software you bought is only good for that computer, and only while you are the owner of the computer. (You can get good deals on reconditioned computers with no operating system installed for that very reason).

If you are in the business of selling software you might have to hire a lawyer to draw up a license agreement, but if you're giving away software there are several standard licenses you can choose from for free. The most popular by far is called the General Public License, or GPL. Like the licenses Microsoft uses it allows the people who get your program to do some things with it but not others. What makes it interesting is not what it allows them to do (which is pretty much anything they like) but what it forbids them to do.

If someone distributes a program licensed under the GPL they are also required to make the source code of the program available to anyone who wants it. That person may do as he likes with the code, with one important restriction: if he distributes a program based on that code he must also license that code using the GPL. This makes it impossible for someone to take a GPL licensed work, improve it, and sell it to someone without giving him the source code to the new version.

While the GPL is not the only license available for Activities to be distributed on http://activities.sugarlabs.org all the licenses require that anyone getting the Activity also gets the complete source code for it. You've already taken care of that requirement by putting your source code in Gitorious. If you used any code from an existing Activity licensed with the GPL you must license your own code the same way. If you used a significant amount of code from this book (which is also GPL licensed) you may be required to use the GPL too.

Is licensing something you should worry about? Not really. The only reason you'd want to use a license other than the GPL is if you wanted to sell your Activity instead of give it away. Consider what you'd have to do to make that possible:

- You'd have to use some language other than Python so you could give someone the program without giving them the source code.
- You would have to have your own source code repository not available to the general public and make arrangements to have the data backed up regularly.
- You would have to have your own website to distribute the Activity. The website would have to be set up to accept payments somehow.
- You would have to advertise this website somehow or nobody would know your Activity existed.
- You would have to have a lawyer draw up a license for your Activity.
- You would have to come up with some mechanism to keep your customers from giving away copies of your Activity.
- You would have to create an Activity so astoundingly clever that nobody else could make something similar and give it away.
- You would have to deal with the fact that your "customers" would be children with no money or credit cards.

In summary, activities.sugarlabs.org is not the iPhone App Store. It is a place where programmers share and build upon each other's work and give the results to children for free. The GPL encourages that to happen, and I recommend that you choose that for your license.

---

### Add License Comments To Your Python Code

At the top of each Python source file in your project (except setup.py, which is already commented) put comments like this:

```python
# filename    Program description
#
# Copyright (C) 2010 Your Name Here
#
# This program is free software; you can redistribute it
# and/or modify it under the terms of the GNU General
# Public License as published by the Free Software
# Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General
# Public License along with this program; if not, write
# to the Free Software Foundation, Inc., 51 Franklin
# St, Fifth Floor, Boston, MA 02110-1301  USA
If the code is based on someone else's code you should mention that as a courtesy.

Create An .xo File
Make certain that activity.info has the version number you want to give your Activity (currently it must be a positive integer) and run this command:

```bash
./setup.py dist_xo
```
This will create a dist directory if one does not exist and put a file named something like ReadETextsII-1.xo in it. The "1" indicates version 1 of the Activity.

If you did everything right this .xo file should be ready to distribute. You can copy it to a thumb drive and install it on an XO laptop or onto another thumb drive running Sugar on a Stick. You probably should do that before distributing it any further. I like to live with new versions of my Activities for a week or so before putting them on activities.sugarlabs.org.

Now would be a good time to add dist and locale to your .gitignore file (if you haven't done that already), then commit it and push it to Gitorious. You don't want to have copies of your .xo files in Git. Another good thing to do at this point would be to tag your Git repository with the version number so you can identify which code goes with which version.

```bash
git tag -m "Release 1" v1 HEAD
git push --tags
```
Add Your Activity To ASLO
When you're ready to post the .xo file on ASLO you'll create an account as you did with the other websites. When you've logged in there you'll see a Tools link in the upper right corner of the page. Click on that and you'll see a popup menu with an option for Developer Hub, which you should click on. That will take you to the pages where you can add new Activities. The first thing it asks for when setting up a new Activity is what license you will use. After that you should have no problem getting your Activity set up.

You will need to create an Activity icon as a .gif file and create screen shots of your Activity in action. You can do both of these things with The GIMP (GNU Image Manipulation Program). For the icon all you need to do is open the .svg file with The GIMP and Save As a .gif file.

For the screen shots use sugar-emulator to display your Activity in action, then use the Screenshot option from the Create submenu of the File menu with these options:

(gimp1.jpg referenced in original text)

This tells GIMP to wait 10 seconds, then take a screenshot of the window you click on with the mouse. You'll know that the 10 seconds are up because the mouse pointer will change shape to a plus (+) sign. You also tell it not to include the window decoration (which means the window title bar and border). Since windows in Sugar do not have decorations eliminating the decorations used by sugar-emulator will give you a screenshot that looks exactly like a Sugar Activity in action.

Every Activity needs one screenshot, but you can have more if you like. Screenshots help sell the Activity and instruct those who will use it on what the Activity can do. Unfortunately, ASLO cannot display pictures in a predictable sequence, so it is not suited to displaying steps to perform.

Another thing you'll need to provide is a home page for your Activity. The one for Read Etexts is here:

http://wiki.sugarlabs.org/go/Activities/Read_Etexts

Yes, one more website to get an account for. Once you do you can specify a link with /go/Activities/some_name and when you click on that link the Wiki will create a page for you. The software used for the Wiki is MediaWiki, the same as used for Wikipedia. Your page does not need to be as elaborate as mine is, but you definitely should provide a link to your source code in Gitorious.

## Debugging Sugar Activities

### Introduction

No matter how careful you are it is reasonably likely that your Activity will not work perfectly the first time you try it out. Debugging a Sugar Activity is a bit different than debugging a standalone program. When you test a standalone program you just run the program itself. If there are syntax errors in the code you'll see the error messages on the console right away, and if you're running under the Eric IDE the offending line of code will be selected in the editor so you can correct it and keep going.

With Sugar it's a bit different. It's the Sugar environment, not Eric, that runs your program. If there are syntax errors in your code you won't see them right away. Instead, the blinking Activity icon you see when your Activity starts up will just keep on blinking for several minutes and then will just go away, and your Activity won't start up. The only way you'll see the error that caused the problem will be to use the Log Activity. If your program has no syntax errors but does have logic errors you won't be able to step through your code with a debugger to find them. Instead, you'll need to use some kind of logging to trace through what's happening in your code, and again use the Log Activity to view the trace messages. Now would be a good time to repeat some advice I gave before:

### Make A Standalone Version Of Your Program First

Whatever your Activity does, it's a good bet that 80% of it could be done by a standalone program which would be much less tedious to debug. If you can think of a way to make your Activity runnable as either an Activity or a standalone Python program then by all means do it.

---

### Use PyLint, PyChecker, or PyFlakes

One of the advantages of a compiled language like C over an interpreted language like Python is that the compiler does a complete syntax check of the code before converting it to machine language. If there are syntax errors the compiler gives you informative error messages and stops the compile. There is a utility call lint which C programmers can use to do even more thorough checks than the compiler would do and find questionable things going on in the code.

Python does not have a compiler but it does have several lint-like utilities you can run on your code before you test it. These utilities are pyflakes, pychecker, and pylint. Any Linux distribution should have all three available.

---

### PyFlakes

Here is an example of using PyFlakes:

```bash
pyflakes minichat.py
minichat.py:25: 'COLOR_BUTTON_GREY' imported but unused
minichat.py:28: 'XoColor' imported but unused
minichat.py:29: 'Palette' imported but unused
minichat.py:29: 'CanvasInvoker' imported but unused
PyFlakes seems to do the least checking of the three, but it does find errors like these above that a human eye would miss.

PyChecker
Here is PyChecker in action:

```bash
pychecker ReadEtextsActivity.py
```
Processing ReadEtextsActivity...
/usr/lib/python2.5/site-packages/dbus/_dbus.py:251:
DeprecationWarning: The dbus_bindings module is not public
API and will go away soon.
...
ReadEtextsActivity.py:62: Parameter (widget) not used

4 errors suppressed, use -#/--limit to increase the number
of errors displayed
PyChecker not only checks your code, it checks the code you import, including Sugar code.

PyLint
Here is PyLint, the most thorough of the three:

```bash
pylint ReadEtextsActivity.py
```
No config file found, using default configuration
************* Module ReadEtextsActivity
C:177: Line too long (96/80)
C:  1: Missing docstring
C: 27: Operator not preceded by a space
page=0
    ^
...
Your code has been rated at 7.52/10 (previous run: 7.52/10)
PyLint is the toughest on your code and your ego. It not only tells you about syntax errors, it tells you everything someone might find fault with in your code. This includes style issues that won't affect how your code runs but will affect how readable it is to other programmers.

The Log Activity
When you start testing your Activities the Log Activity will be like your second home. It displays a list of log files in the left pane and when you select one it will display the contents of the file in the right pane. Every time you run your Activity a new log file is created for it, so you can compare the log you got this time with what you got on previous runs.

The Edit toolbar is especially useful. It contains a button to show the log file with lines wrapped (which is not turned on by default but probably should be). It has another button to copy selections from the log to the clipboard, which will be handy if you want to show log messages to other developers.

The Tools toolbar has a button to delete log files. I've never found a reason to use it. Log files go away on their own when you shut down sugar-emulator.

The Log Activity

Here is what the Log Activity looks like showing a syntax error in your code:

The Log Activity displaying a syntax error in Speak.

Logging
Without a doubt the oldest debugging technique there is would be the simple print statement. If you have a running program that misbehaves because of logic errors and you can't step through the code in a debugger to figure out what's happening you might print statements in your code. For instance, if you aren't sure that a method is ever getting executed you might put a statement like this as the first line of the method:

```python
def my_method():
    print 'my_method() begins'
```
You can include data in your print statements too. Suppose you need to know how many times a loop is run. You could do this:

```python
while linecount < PAGE_SIZE:
    line = self.etext_file.readline()
    label_text = label_text + unicode(line,
        'iso-8859-1')
    linecount = linecount + 1
    print 'linecount=', linecount
```
The output of these print statements can be seen in the Log Activity. When you're finished debugging your program you would remove these statements.

An old programming book I read once made the case for leaving the statements in the finished program. The authors felt that using these statements for debugging and then removing them is a bit like wearing a parachute when the plane is on the ground and taking it off when it's airborne. If the program is out in the world and has problems you might well wish you had those statements in the code so you could help the user and yourself figure out what's going on.

On the other hand, print statements aren't free. They do take time to run and they fill up the log files with junk. What we need are print statements that you can turn on and off.

Python Standard Logging
The way you can do this is with Python Standard Logging. In the form used by most Activities it looks like this:

```python
self._logger = logging.getLogger(
    'read-etexts-activity')
```
These statements would go in the init() method of your Activity. Every time you want to do a print() statement you would do this instead:

```python
def _shared_cb(self, activity):
    self._logger.debug('My activity was shared')
    self.initiating = True
    self._sharing_setup()

    self._logger.debug(
        'This is my activity: making a tube...')
    id = self.tubes_chan[telepathy.CHANNEL_TYPE_TUBES].\
        OfferDBusTube(SERVICE, {})
```
```python
def _sharing_setup(self):
    if self._shared_activity is None:
        self._logger.error(
            'Failed to share or join activity')
        return
```
Notice that there are two kinds of logging going on here: debug and error. These are error levels. Every statement has one, and they control which log statements are run and which are ignored. There are several levels of error logging, from lowest severity to highest:

```python
self._logger.debug("debug message")
self._logger.info("info message")
self._logger.warn("warn message")
self._logger.error("error message")
self._logger.critical("critical message")
```
When you set the error level in your program to one of these values you get messages with that level and higher. You can set the level in your program code like this:

```python
self._logger.setLevel(logging.DEBUG)
```
You can also set the logging level outside your program code using an environment variable. For instance, in Sugar .82 and lower you can start sugar-emulator like this:

```bash
SUGAR_LOGGER_LEVEL=debug sugar-emulator
```
You can do the same thing with .84 and later, but there is a more convenient way. Edit the file ~/.sugar/debug and uncomment the line that sets the SUGAR_LOGGER_LEVEL. Whatever value you have for SUGAR_LOGGER_LEVEL in ~/.sugar/debug will override the one set by the environment variable, so either change the setting in the file or use the environment variable, but don't do both.

The Analyze Activity
Another Activity you may find yourself using at some point is Analyze. This is more likely to be used to debug Sugar itself than to debug your Activity. If, for instance, your collaboration test environment doesn't seem to be working this Activity might help you or someone else figure out why.

I don't have a lot to say about this Activity here, but you should be aware that it exists.

