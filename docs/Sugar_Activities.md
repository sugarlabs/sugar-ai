# Sugar Activities: Complete Guide

**Source:** FLOSS Manuals – “Make Your Own Sugar Activities”  
**Authors:** James D. Simmons and contributors  
**License:** GNU General Public License v2 or later (GPL-2.0-or-later)  
**Original source URL:** https://archive.flossmanuals.net/make-your-own-sugar-activities/


## INTRODUCTION

# Introduction

"This book is a record of a pleasure trip. If it were a record of a solemn scientific expedition, it would have about it that gravity, that profundity, and that impressive incomprehensibility which are so proper to works of that kind, and withal so attractive."

From the Preface to The Innocents Abroad , by Mark Twain

The purpose of this book is to teach you what you need to know to write Activities for Sugar, the operating environment developed for the One Laptop Per Child project. This book does not assume that you know how to program a computer, although those who do will find useful information in it.Â  My primary goal in writing it is to encourage non programmers, including children and their teachers, to create their own Sugar Activities.Â  Because of this goal I will include some details that other books would leave out and leave out things that others would include.Â  Impressive incomprehensibility will be kept to a minimum.

If you just want to learn how to write computer programs Sugar provides many Activities to help you do that: Etoys, Turtle Art, Scratch, and Pippy. None of these are really suitable for creating Activities so I won't cover them in this book, but they're a great way to learn about programming. If you decide after playing with these that you'd like to try writing an Activity after all you'll have a good foundation of knowledge to build on.

When you have done some programming then you'll know how satisfying it can be to use a program that you made yourself, one that does exactly what you want it to do.Â  Creating a Sugar Activity takes that enjoyment to the next level.Â  A useful Sugar Activity can be translated by volunteers into every language, be downloaded hundreds of times a week and used every day by students all over the world.

A book that teaches everything you need to know to write Activities would be really, really long and would duplicate material that is already available elsewhere. Because of this, I am going to write this as sort of a guided tour of Activity development. That means, for example, that I'll teach you what Python is and why it's important to learn it but I won't teach you the Python language itself. There are excellent tutorials on the Internet that will do that, and I'll refer you to those tutorials.

There is much sample code in this book, but there is no need for you to type it in to try it out.Â  All of the code is in a Git repository that you can download to your own computer.Â  If you've never used Git there is a chapter that explains what it is and how to use it.

I started writing Activities shortly after I received my XO laptop. When I started I didn't know any of the material that will be in this book. I had a hard time knowing where to begin. What I did have going for me though was a little less than 30 years as a professional programmer. As a result of that I think like a programmer. A good programmer can take a complex task and divide it up into manageable pieces. He can figure out how things must work, and from that figure out how they do work. He knows how to ask for help and where. If there is no obvious place to begin he can begin somewhere and eventually get where he needs to go.

Because I went through this process I think I can be a pretty good guide to writing Sugar Activities.Â  Along the way I hope to also teach you how to think like a programmer does.

From time to time I may add chapters to this book.Â  Sugar is a great application platform and this book can only begin to tell you what is possible.

In the first edition of this book I expressed the wish that future versions of the book would have guest chapters on more advanced topics written by other experienced Activity developers.Â  That wish has been realized.Â  Not only does the book have guest chapters, but some of the new content has been created by young developers from the Google Code-in of 2012, for which I was one of the mentors.Â  Their bios are in the About The Authors chapter.Â  Read them and prepare to be impressed!

This book eliminates some content that was part of the first edition. The current version of Sugar uses GTK 3, and most of the examples in this addition use GTK 3 as well. In the first edition all the examples used GTK 2. The first edition also explained how to support multiple versions of Sugar with one Activity. That is no longer relevant to most Activity developers, so it has been removed.

If you need the content of the first edition you will find it at archive.org . Code examples from the first edition are also still available at git.sugarlabs.org .

In addition to examples using GTK3 this edition introduces the use of HMTL 5 and WebKit to create Activities. The advantage of HTML 5 is that it can run on Android and other tablet devices. There has been a fair amount of interest in using these devices for education, and if you develop using HTML 5 you can make versions of your Activity for both Sugar and Android.

## Formats For This Book

This book is part of the FLOSS Manuals project and is available for online viewing at their website:

http://en.flossmanuals.net/

At this time this is the only place the second edition is available, but it will in time be made available everywhere the first edition is.

You will can purchase a printed and bound version of the first edition of book at Amazon.com :

http://www.amazon.com/James-Simmons/e/B005197ZTY/ref=ntt_dp_epwbk_0

The Internet Archive has the first edition of this book available as a full color PDF, as well as EPUB, MOBI, and DjVu versions, all of which you can download for free:

http://www.archive.org/details/MakeYourOwnSugarActivities Â

The Amazon Kindle Store has exactly the same MOBI version as the Internet Archive does.

If you choose to read this book on a Kindle or a Nook be aware that these devices have narrow screens not well suited for displaying program listings.Â  I suggest you refer to the FLOSS Manuals website to see what the code looks like properly formatted.

There is a Spanish version of the first edition of this book,Â CÃ³mo Hacer Una Actividad Sugar , which is available in all the same places as this version.

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## WHAT IS SUGAR?

# What is Sugar?

Sugar is the user interface designed for the XO laptop. It can now be installed on most PCs, including older models that can't run the latest Windows software. You can also install it on a thumb drive (Sugar on a Stick) and boot your PC from that.

When the XO laptop first came out some people questioned the need for a new user interface. Wouldn't it be better for children to learn something more like what they would use as adults? Why not give them Microsoft Windows instead?

This would be a reasonable question if the goal was to train children to use computers and nothing else. It would be even more reasonable if we could be sure that the software they would use as adults looked and worked like the Microsoft Windows of today.Â  These are of course not reasonable assumptions.

The OLPC project is not just about teaching computer literacy. It is about teaching everything : reading, writing, arithmetic, history, science, arts and crafts, computer programming, music composition, and everything else. Not only do we expect the child to use the computer for her school work, we expect her to take it home and use it for her own explorations into subjects that interest her.Â

This is a great deal more than anyone has done with computers for education, so it is reasonable to rethink how children should work with computers. Sugar is the result of that rethinking.

Sugar has the following unique features:

## The Journal

The Journal is where all the student's work goes. Instead of files and folders there is a list of Journal entries. The list is sorted in descending order by the date and time it was last worked on. In a way it's like the "Most Recently Used" document menu in Windows, except instead of containing just the last few items it contains everything and is the normal way to save and resume work on something.

The Journal makes it easy to organize your work.Â  Any work you do is saved to the Journal.Â  Anything you download from the web goes in the Journal.Â  If you've ever downloaded a file using a web browser, then had to look for it afterwards because it went in some directory other than the one you expected, or if you ever had to help your parents when they were in a similar situation, you can understand the value of the Journal.

The Journal has metadata for each item in it. Metadata is information about information. Every Journal entry has a title, a description, a list of keywords, and a screen shot of what it looked like the last time it was used. It has an activity id that links it to the Activity that created it, and it may have a MIME type as well (which is a way of identifying Journal entries so that items not created by an Activity may still be used by an Activity that supports that MIME type).

In addition to these common metadata items a Journal entry may be given custom metadata by an Activity. For instance, the Read Activity uses custom metadata to save the page number you were reading when you quit the Activity. When you resume reading later the Activity will put you on that page again.

In addition to work created by Activities, the Journal can contain Activities themselves. To install an Activity you can use the Browse Activity to visit the website http://activities.sugarlabs.org and download it. It will automatically be saved to the Journal and be ready for use. If you don't want the Activity any more, simply delete it from the Journal and it's completely gone . No uninstall programs, no dialog boxes telling you that such and such a .DLL doesn't seem to be needed anymore and do you want to delete it? No odd bits and pieces left behind.

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

The Friends View is similar to the Neighborhood View, but only contains icons for people you have designated as Friends. Collaboration can be offered at three levels: with individual persons, with the whole Neighborhood, and with Friends. Note that the student alone decides who her Friends are. There is no need to ask to be someone's Friend.Â  It's more like creating a mailing list in email.

## Security

Protecting computers from malicious users is very important, and if the computers belong to students it is doubly important. It is also more difficult, because we can't expect young students to remember passwords and keep them secret. Since Sugar runs on top of Linux viruses aren't much of a problem, but malicious Activities definitely are. If an Activity was allowed unrestricted access to the Journal, for instance, it could wipe it out completely. Somebody could write an Activity that seems to be harmless and amusing, but perhaps after some random number of uses it could wipe out a student's work.

The most common way to prevent a program from doing malicious things is to make it run in a sandbox. A sandbox is a way to limit what a program is allowed to do. With the usual kind of sandbox you either have an untrusted program that can't do much of anything or a trusted program that is not restricted at all. An application becomes trusted when a third party vouches for it by giving it a signature . The signature is a mathematical operation done on the program that only remains valid if the program is not modified.

Sugar has a more sophisticated sandbox for Activities than that. No Activity needs to be trusted or is trusted. Every Activity can only work with the Journal in a limited, indirect way. Each Activity has directories specific to it that it can write to, and all other directories and files are limited to read-only access. In this way no Activity can interfere with the workings of any other Activity. In spite of this, an Activity can be made to do what it needs to do.

## Summary

Sugar is an operating environment designed to support the education of children. It organizes a child's work without needing files and folders. It supports collaboration between students. Finally, it provides a robust security model that prevents malicious programs from harming a student's work.

It would not be surprising to see these features someday adopted by other desktop environments.

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## WHAT IS A SUGAR ACTIVITY?

# What is a Sugar Activity?

A Sugar Activity is a self-contained Sugar application packaged in a .xo bundle.

An .xo bundle is an archive file in the Zip format.Â  It contains:

- A MANIFEST file listing everything in the bundle
- An activity.info file that has attributes describing the Activity as name=value pairs.Â  These attributes include the Activity name, its version number, an identifier, and other things we will discuss when we write your first Activity.
- An icon file (in SVG format)
- Files containing translations of the text strings the Activity uses into many languages
- The program code to run the Activity
A Sugar Activity will generally have some Python code that extends a Python class called Activity.Â  It may also make use of code written in other languages if that code is written in a way that allows it to be used from Python (this is called having Python bindings ).Â  For instance, you can use WebKit, an HTML component, as part of your Activity. This would enable you to write most of your Activity using HTML 5 instead of Python.

It is even possible to write a Sugar Activity without using Python at all, but this is beyond the scope of this book.

There are only a few things that an Activity can depend on being included with every version of Sugar. These include modules like Evince (PDF and other document viewing), WebKit (rendering web pages), and Python libraries like PyGTK and PyGame.Â  Everything needed to run the Activity that is not supplied by Sugar must go in the bundle file.Â  A question sometimes heard on the mailing lists is "How do I make Sugar install X the first time my Activity is run?"Â  The answer: you don't.Â  If you need X it needs to go in the bundle.Â

You can install an Activity by copying or downloading it to the Journal. You uninstall it by removing it from the Journal. There is no Install Shield to deal with, no deciding where you want the files installed, no possibility that installing a new Activity will make an already installed Activity stop working. For the child installing an Activity the process is no more difficult than installing an app on a smart phone.

An Activity generally creates and reads objects in the Journal.Â  A first rate Activity will provide some way for the Activity to be shared by multiple users.

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?

# What Do I Need To Know To Write A Sugar Activity?Â

If you are going to write Sugar Activities you should learn something about the topics described in this chapter. There is no need to become an expert in any of them, but you should bookmark their websites and skim through their tutorials. This will help you to understand the code samples we'll be looking at.

## Python

Python is the most used language for writing Activities.Â  While you can use other languages, most Activities have at least some Python in them.Â  Sugar provides a Python API that simplifies creating Activities.Â  While it is possible to write Activities using no Python at all (like Etoys ), it is unusual.Â

Most of the examples in this book are written entirely in Python. Â In a later Guest Chapter you'll see how you can mix Python and HTML 5 to make an impressive Activity.

There are compiled languages and interpreted languages. In a compiled language the code you write is translated into the language of the chip it will run on and it is this translation that is actually run by the OS. In an interpreted language there is a program called an interpreter that reads the code you write and does what the code tells it to do. (This is over simplified, but close enough to the truth for this chapter).

Python is an interpreted language. There are advantages to having a language that is compiled and there are advantages to having an interpreted language. The advantages Python has for developing Activities are:

- It is portable. In other words, you can make your program run on any chip and any OS without making a version specific to each one. Compiled programs only run on the OS and chip they are compiled for.
- Since the source code is the thing being run, you can't give someone a Python program without giving them the source code. You can learn a lot about Activity programming by studying other people's code, and there is plenty of it to study.
- It is an easy language for new programmers to learn, but has language features that experienced programmers need.
- It is widely used. One of the best known Python users is Google. They use it enough that they have a project named âUnladen Swallowâ to make Python programs run faster.
The big advantage of a compiled language is that it can run much faster than an interpreted language. However, in actual practice a Python program can perform as well as a compiled program. To understand why this is you need to understand how a Python program is made.

Python is known as a âglueâ language. The idea is that you have components written in various languages (usually C and C++) and they have Python bindings. Python is used to âglueâ these components together to create applications. In most applications the bulk of the application's function is done by these compiled components, and the application spends relatively little time running the Python code that glues the components together.

In addition to Activities using Python most of the Sugar environment itself is written in Python.

If you have programmed in other languages before there is a good tutorial for learning Python at the Python website: http://docs.python.org/tutorial/ .Â  If you're just starting out in programming you might check out Invent Your Own Computer Games With Python , which you can read for free at http://inventwithpython.com/ .

## PyGTK

GTK+ is a set of components for creating user interfaces. These components include things like buttons, scroll bars, list boxes, and so on. It is used by GNOME desktop environment and the applications that run under it. Sugar Activities use a special GNOME theme that give GTK+ controls a unique look.

PyGTK is a set of Python bindings that let you use GTK+ components in Python programs. Sugar 3 uses GTK+ 3 and Activities written for it should also, although the previous version of PyGTK still works. GTK+3 is very different from what went before. There is a tutorial showing how to use it at the PyGTK website: http://python-gtk-3-tutorial.readthedocs.org/en/latest/. You'll also find an article on converting Activities using GTK 2 to use GTK 3 in the guest chapters section of this book.

## PyGame

The alternative to using PyGTK for your Activity is PyGame. PyGame can create images called sprites and move them around on the screen. As you might expect, PyGame is mostly used for writing games. It is less commonly used in Activities than PyGTK.

The tutorial to learn about PyGame is at the PyGame website: http://www.pygame.org/wiki/tutorials . The website also has a bunch of pygame projects you can download and try out.

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS


## PROGRAMMING

## SETTING UP A DEVELOPMENT ENVIRONMENT

# Setting Up a Sugar Development Environment

It is not currently practical to develop Activities for the XO on the XO. It's not so much that you can't do it, but that it's easier and more productive to do your development and testing on another machine running a more conventional environment. This gives you access to better tools and it also enables you to simulate collaboration between two computers running Sugar using only one computer.

## Installing Linux

If you're going to develop Sugar Activities you're going to need a machine that runs Linux. Â While you can write programs using Python, PyGTK, and PyGame on any operating system you're going to need Linux to make them into Sugar Activities and test them under Sugar. It is possible to write a standalone Python program in Windows and then turn over the code to another developer who runs Linux who would then make an Activity out of it. If you're just starting out in computer programming that may be an attractive option.

Installing Linux is not the test of manhood it once was. Anyone can do it. The GNOME desktop provided with Linux is very much like Windows so you'll feel right at home using it.

When you install Linux you have the option to do a dual boot, running Linux and Windows on the same computer (but not at the same time). This means you set aside a disk partition for use by Linux and when you start the computer a menu appears asking which OS you want to start up. The Linux install will even create the partition for you, and a couple of gigabytes is more than enough disk space. Sharing a computer with a Linux installation will not affect your Windows installation at all.Â

Sugar Labs has been working to get Sugar included with all Linux distributions. If you already have a favorite distribution, chances are the latest version of it includes Sugar. Fedora, openSuse, Debian, and Ubuntu all include Sugar. If you already use Linux, see if Sugar is included in your distribution. If not, Fedora is what is used by the XO computer so a recent version of Fedora might be your best bet. You can download the Fedora install CD or DVD here:Â https://fedoraproject.org/get-fedora .

It is worth pointing out that all of the other tools I'm recommending are included in every Linux distribution, and they can be installed with no more effort than checking a check box. The same tools often will run on Windows, but installing them there is more work than you would expect for Windows programs.

There is another way to install Linux which is to use a virtual machine.Â A virtual machine is a way to run one operating system on top of another one. The operating system being run is fooled into thinking it has the whole computer to itself. (Computer industry pundits will tell you that using virtual machines is the newest new thing out there. Old timers like me know that IBM was doing it on their mainframe computers back in the 1970's).

If you're used to Windows you might think that running Linux in a VM from Windows instead of installing Linux might be the easier option. In practice it is not. Linux running in a VM is still Linux, so you're still going to have to learn some things about Linux to do Activity development. Also, running a second OS in a VM requires a really powerful machine with gigabytes of memory. On the other hand, I did my first Sugar development using Linux on an IBM NetVista Pentium IV I bought used for a little over a hundred dollars, shipping included. It was more than adequate, and would still be adequate today.

Having said that, there are reasons to like the virtual machine approach. I have used this method to run Linux on a Macintosh. Â It looks like this:

It is really fast and easy to set this up if you have a recent Mac. Â The screenshot is from a Mac Mini my wife and I bought recently which has 2 gig of RAM and more disk space than we'll ever use. Â  All I needed to do was download the Mac OS version fromÂ http://www.virtualbox.org , then download a Fedora image fromÂ http://virtualboxes.org/images/fedora/ . Â This image needs to be unpacked before you can use it, so get Stuffit Expander for free from the Apple Market. Â Run Virtual Box, configure your unpacked image and in no time flat you have Fedora running in a window on your desktop. Â You can go to Add/Remove Programs and add Sugar and your favorite development tools and you're set.

One of the neat things about Virtual Box images is you can have more than one. The Virtual Boxes website has images from Fedora 10 up to the latest, so I can have a second image running Fedora 10 to test compatibility with very old versions of Sugar that may still be in use in the field. Â I once set up a machine running Fedora 10 and deliberately avoided upgrading just so I could do this kind of testing. Â Now that I have the Mac Mini running Virtual Box I should be able to give that box an upgrade. Being able to run multiple versions of Linux on the same box easily is really the biggest advantage of the Virtual Box method.

Virtual Box will also run on Windows if you have a powerful enough machine.

As for the Macintosh, it should also be possible to install Fedora Linux on an Intel Macintosh as a dual boot, just like you do with Windows.Â  Check the Fedora website for details.

What About Using sugar-build?

Sugar-build is a script that downloads the source code for the latest version of all the Sugar modules and compiles it into a subdirectory of your home directory.Â  It doesn't actually install Sugar on your system.Â  Instead, you run it out of the directory you installed it in.Â  Because of the way it is built and run it doesn't interfere with the modules that make up your normal desktop. If you are developing Sugar itself, or if you are developing Activities that depend on the very latest Sugar features you'll need to run sugar-build.

Running this script is a bit more difficult than just installing the Sugar packages that come with the distribution.Â  You'll need to install Git, run a Git command from the terminal to download the sugar-build scripts, then do a make run which will download more code, possibly ask you to install more packages, and ultimately compile everything and run it. Â When you're done you'll have an up to date test environment that you can run as an alternative to sugar-emulator .Â  There is no need to uninstall sugar-emulator; both can coexist.

The documentation for sugar-build will be found here:

http://developer.sugarlabs.org/

Should you consider using it? The short answer is no. A longer answer is probably not yet .

If you want your Activities to reach the widest possible audience you don't want the latest Sugar. In fact, if you want a test environment that mimics what is on most XO computers right now you might need to use an older version of Fedora. Because updating operating systems in the field can be a major undertaking for a school many XO's will be not be running the very latest Sugar or even the one shipped with the latest Fedora for quite some time.

Strictly speaking sugar-build is just the script that downloads and compiles Sugar.Â  If you wanted to be correct you would say "Run the copy of sugar-runner you made with sugar-build". However, recent Linux distributions like Fedora 20 come with the Sugar environment as something you can install like any other Linux package. There needs to be a way to distinguish between running what came with your distribution and what you download and compile yourself and run out of your home directory. To refer to the second option I'll say "use sugar-build" or "run sugar-build".

## Python

We'll be doing all the code samples in Python so you'll need to have Python installed.Â  Python comes with every Linux distribution.Â  You can download installers for Windows and the Macintosh at http://www.python.org/ .

There are a couple of Python libraries you'll want to have as well: PyGTK and PyGame. Â These come with every Linux distribution. Â If you want to try them out in Windows or on the Macintosh you can get them here:

http://www.pygtk.org/

http://www.pygame.org/news.html Â

## Eric

Developers today expect their languages to be supported by an Integrated Development Environment and Python is no exception. An IDE helps to organize your work and provides text editing and a built in set of programming and debugging tools.

Â

There are two Python IDE's I have tried: Eric and Idle. Eric is the fancier of the two and I recommend it. Every Linux distribution should include it. It looks like it might work on Windows too. You can learn more about it at the Eric website: http://eric-ide.python-projects.org/ .

## SPE (Stani's Python Editor)

This is an IDE I discovered while writing this book.Â  It comes with Fedora and in addition to being a Python editor it will make UML diagrams of your code and show PyDoc for it.Â  Here is SPE showing a UML diagram for one of the Activities in this book:

If you're an experienced developer you might find this a useful alternative to Eric.Â  If you're just starting out Eric should meet your needs pretty well.

## Other IDE's

There is also a commercial Python IDE called Wingware, which has a version you can use for free.Â  You can learn more about it at http://www.wingware.com/ .

## Inkscape

Inkscape is a tool for creating images in SVG format. Sugar uses SVG for Activity icons and other kinds of artwork. The âXOâ icon that represents each child in the Neighborhood view is an SVG file that can be modified.

Â

Inkscape comes with every Linux distribution, and can be installed on Windows as well. You can learn more about it here: http://www.inkscape.org/ .

## Git

Git is a version control system. It stores versions of your program code in a way that makes them easy to get back. Whenever you make changes to your code you ask Git to store your code in its repository. If you need to look at an old version of that code later you can. Even better, if some problem shows up in your code you can compare your latest code to an old, working version and see exactly what lines you changed.

Â â

If there are two people working on the same program independently a version control system will merge their changes together automatically.

Suppose you're working on a major new version of your Activity when someone finds a really embarrassing bug in the version you just released. If you use Git you don't need to tell people to live with it until the next release, which could be months away. Instead you can create a branch of the previous version and work on it alongside the version you're enhancing. In effect Git treats the old version you're fixing and the version you're improving as two separate projects.

You can learn more about Git at the Git website: http://git-scm.com/ .

When you're ready for a Git repository for your project you can set one up here: http://git.sugarlabs.org/ .Â  I will have more to say about setting up and using a Git repository later in this book.

There is a Git repository containing all the code examples from this book.Â  Once you have Git installed you can copy the repository to your computer with this command:

This command should be typed all on one line.Â  The backslash (\) character at the end of the first line is used in Linux to continue a long command to a second line.Â  It is used here to make the command fit on the page of the printed version of this book.Â  When you type in the command you can leave it out and type myo-sugar-activities-examples/mainline.git immediately following git.sugarlabs.org/ .

This convention of splitting long commands over multiple lines will be used throughout this book.Â Â  In addition to that, the code in Git will generally have longer lines than you'll see in the code listings in the book.Â  For that reason I'd recommend that you not try to type in the code from these listings, but use the code in Git instead.

## The GIMP

The GIMP is one of the most useful and badly named programs ever developed. You can think of it as a free version of Adobe Photoshop. If you need to work with image files (other than SVG's) you need this program.

Â

You may never need this program to develop the Activity itself, but when it's time to distribute the Activity you'll use it to create screen shots of your Activity in action. Nothing sells an Activity to a potential user like good screen shots.

## Sugar Emulation

Most Linux distributions should have Sugar included. In Fedora you can run Sugar as an alternative desktop environment. When you log in to GDM Sugar appears as a desktop selection alongside GNOME, KDE, Window Maker, and any other window managers you have installed.

This is not the normal way to use Sugar for testing. The normal way uses a tool called Xephyr to run a Sugar environment in a window on your desktop. In effect, Xephyr runs an X session inside a window and Sugar runs in that. You can easily take screen shots of Sugar in action, stop and restart Sugar sessions without restarting the computer, and run multiple copies of Sugar to test collaboration.

Â

I'll have more to say about this when it's time to test your first Activity.Â

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## CREATING YOUR FIRST ACTIVITY

# Creating your First Sugar Activity

## Make A Standalone Python Program First

The best advice I could give a beginning Activity developer is to make a version of your Activity that can run on its own, outside of the Sugar environment. Testing and debugging a Python program that stands alone is faster, easier and less tedious than doing the same thing with a similar Activity. You'll understand why when you start testing your first Activity.

The more bugs you find before you turn your code into an Activity the better. In fact, it's a good idea to keep a standalone version of your program around even after you have the Activity version well underway. I used my standalone version of Read Etexts to develop the text to speech with highlighting feature. This saved me a lot of time, which was especially important because I was figuring things out as I went.

Our first project will be a version of the Read Etexts Activity I wrote.

## Inherit From The sugar.activity.Activity Class

Next we're going to take our standalone Python program and make an Activity out of it.Â  To do this we need to understand the concept of inheritance .Â  In everyday speech inheritance means getting something from your parents that you didn't work for.Â  A king will take his son to a castle window and say, "Someday, lad, this will all be yours!"Â  That's inheritance.

## Package The Activity

Now we have to package up our code to make it something that can be run under Sugar and distributed as an .xo file.Â  This involves setting up a MANIFEST, activity.info, setup.py, and creating a suitable icon with Inkscape.

## Add Refinements

Every Activity will have the basic Activity toolbar. For most Activities this will not be enough, so we'll need to create some custom toolbars as well. Then we need to hook them up to the rest of the Activity code so that what happens to the toolbar triggers actions in the Activity and what happens outside the toolbar is reflected in the state of the toolbar.

In addition to toolbars we'll look at some other ways to spiff up your Activity.

## Put The Project Code In Version Control

By this time we'll have enough code written that it's worth protecting and sharing with the world.Â  To do that we need to create a Git repository and add our code to it.Â  We'll also go over the basics of using Git.

## Going International With Pootle

Now that our code is in Git we can request help from our first collaborator: the Pootle translation system.Â  With a little setup work we can get volunteers to make translated versions of our Activity available.

## Distributing The Activity

In this task we'll take our Activity and set it up on http://activities.sugarlabs.org Â  plus we'll package up the source code so it can be included in Linux distributions.

## Add Collaboration

Next we'll add code to share e-books with Friends and the Neighborhood.

## Add Text To Speech

Text to Speech with word highlighting is next.Â  Our simple project will become a Kindle-killer!

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## A STANDALONE PYTHON PROGRAM FOR READING ETEXTS

# A Standalone Python Program For Reading Etexts

## The Program

Our example program is based on the first Activity I wrote, Read Etexts .Â  This is a program for reading free e-books.

The oldest and best source of free e-books is a website called Project Gutenberg ( http://www.gutenberg.org/wiki/Main_Page ).Â  They create books in plain text format, in other words the kind of file you could make if you typed a book into Notepad and hit the Enter key at the end of each line.Â  They have thousands of books that are out of copyright, including some of the best ever written.Â  Before you read further go to that website and pick out a book that interests you.Â  Check out the "Top 100" list to see the most popular books and authors.

The program we're going to create will read books in plain text format only.

There is a Git repository containing all the code examples in this book.Â  Once you have Git installed you can copy the repository to your computer with this command:

The code for our standalone Python program will be found in the directory Make_Standalone_Python_gtk3 in a file named ReadEtexts.py .Â  It looks like this:

## Running The Program

To run the program you should first make it executable.Â  You only need to do this once:

For this example I downloaded the file for Pride and Prejudice .Â  The program will work with either of the Plain text formats, which are either uncompressed text or a Zip file.Â  The zip file is named 1342.zip , and we can read the book by running this from a terminal:

This is what the program looks like in action:

You can use the Page Up, Page Down, Up, Down, Left , and Right keys to navigate through the book and the '+' and '-' keys to adjust the font size.

## How The Program Works

This program reads through the text file containing the book and divides it into pages of 45 lines each.Â  We need to do this because the gtk.TextView component we use for viewing the text would need a lot of memory to scroll through the whole book and that would hurt performance.Â  A second reason is that we want to make reading the e-book as much as possible like reading a regular book, and regular books have pages.Â  If a teacher assigns reading from a book she might say "read pages 35-50 for tommorow".Â  Finally, we want this program to remember what page you stopped reading on and bring you back to that page again when you read the book next time.Â  (The program we have so far doesn't do that yet).

To page through the book we use random access to read the file.Â  To understand what random access means to a file, consider a VHS tape and a DVD.Â  To get to a certain scene in a VHS tape you need to go through all the scenes that came before it, in order.Â  Even though you do it at high speed you still have to look at all of them to find the place you want to start watching.Â  This is sequential access .Â  On the other hand a DVD has chapter stops and possibly a chapter menu.Â  Using a chapter menu you can look at any scene in the movie right away, and you can skip around as you like.Â  This is random access, and the chapter menu is like an index .Â  Of course you can access the material in a DVD sequentially too.

We need random access to skip to whatever page we like, and we need an index so that we know where each page begins.Â  We make the index by reading the entire file one line at a time.Â  Every 45 lines we make a note of how many characters into the file we've gotten and store this information in a Python list.Â  Then we go back to the beginning of the file and display the first page.Â  When the program user goes to the next or previous page we figure out what the new page number will be and look in the list entry for that page.Â  This tells us that page starts 4,200 characters into the file.Â  We use seek() on the file to go to that character and then we read 45 lines starting at that point and load them into the TextView.

When you run this program notice how fast it is.Â  Python programs take longer to run a line of code than a compiled language would, but in this program it doesn't matter because the heavy lifting in the program is done by the TextView, which was created in a compiled language.Â  The Python parts don't do that much so the program doesn't spend much time running them.

Sugar uses Python a lot, not just for Activities but for the Sugar environment itself.Â  You may read somewhere that using so much Python is "a disaster" for performance.Â  Don't believe it.

There are no slow programming languages, only slow programmers.

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## PACKAGE THE ACTIVITY

# Package The Activity

## Add setup.py

You'll need to add a Python program called setup.py to the same directory that you Activity program is in.Â  Every setup.py is exactly the same as every other setup.py.Â  The copies in our Git repository look like this:

Be sure and copy the entire text above, including the comments.

The setup.py program is used by sugar for a number of purposes.Â  If you run setup.py from the command line you'll see the options that are used with it and what they do.

We'll be running some of these commands later on.Â  Don't be concerned about the fix_manifest command being obsolete. Earlier versions of the bundle builder used a file named MANIFEST and the current one doesn't need it.Â  It isn't anything you need to worry about.

## Create activity.info

Next create a directory within the one your progam is in and name it activity .Â  Create a file named activity.info within that directory and enter the lines below into it.Â  Here is the one for our first Activity:

This file tells Sugar how to run your Activity.Â  The properties needed in this file are:

## Create An Icon

Next we need to create an icon named read-etexts.svg and put it in the activity subdirectory.Â â We're going to use Inkscape to create the icon.Â  From the New menu in Inkscape select icon_48x48 .Â  This will create a drawing area that is a good size.

You don't need to be an expert in Inkscape to create an icon.Â  In fact the less fancy your icon is the better.Â  When drawing your icon remember the following points:

- Your icon needs to look good in sizes ranging from really, really small to large.
- It needs to be recognizable when its really, really small.
- You only get to use two colors: a stroke color and a fill color.Â  It doesn't matter which ones you choose because Sugar will need to override your choices anyway, so just use black strokes on a white background.
- A fill color is only applied to an area that is contained within an unbroken stroke.Â  If you draw a box and one of the corners doesn't quite connect the area inside that box will not be filled.Â  Free hand drawing is only for the talented.Â  Circles, boxes, and arcs are easy to draw with Inkscape so use them when you can.
- Inkscape will also draw 3D boxes using two point perspective.Â  Don't use them.Â  Icons should be flat images.Â  3D just doesn't look good in an icon.
- Coming up with good ideas for icons is tough.Â  I once came up with a rather nice picture of a library card catalog drawer for Get Internet Archive Books .Â  The problem is, no child under the age of forty has ever seen a card catalog and fewer still understand its purpose.
When you're done making your icon you need to modify it so it can work with Sugar.Â  Specifically, you need to make it show Sugar can use its own choice of stroke color and fill color.Â  The SVG file format is based on XML, which means it is a text file with some special tags in it.Â  This means that once we have finished editing it in Inkscape we can load the file into Eric and edit it as a text file.

I'm not going to put the entire file in this chapter because most of it you'll just leave alone.Â  The first part you need to modify is at the very beginning.

Before:

After:

Now in the body of the document you'll find references to fill and stroke as part of an attribute called style .Â  Every line or shape you draw will have these, like this:

You need to change each one to look like this:

Note that &stroke_color; and &fill_color; both end with semicolons (;), and semicolons are also used to separate the properties for style.Â  Because of this it is an extremely common beginner's mistake to leave off the trailing semicolon because two semicolons in a row don't look right.Â  Be assured that the two semicolons in a row are intentional and absolutely necessary!Â  Second, the value for style should all go on one line .Â  We split it here to make it fit on the printed page; do not split it in your own icon!

## Install The Activity

There's just one more thing to do before we can test our Activity under the Sugar emulator.Â  We need to install it, which in this case means making a symbolic link between the directory we're using for our code in the ~/Activities/ directory.Â  The symbol ~ refers to the "home" directory of the user we're running Sugar under, and a symbolic link is a way to make a file or directory appear to be located in more than one place without copying it.Â  We make this symbolic link by running setup.py again:

## Running Our Activity

Now at last we can run our Activity under Sugar.Â  To do that we need to learn how to run sugar-emulator or sugar-runner . sugar-runner replaces sugar-emulator and is available in Fedora 20 or later. Everything before that uses sugar-emulator.

Fedora puts a Sugar menu item under the Education menu in Xfce or GNOME 2. You can also run it from the command line:

If your screen resolution is smaller than the default size sugar-emulator runs at it will run full screen.Â  This is not convenient for testing, so you may want to specify your own size:

sugar-runner is similar, but it defaults to running in full screen mode if you run it from the menu or without parameters. To get it to run in a window you need to specify a --resolution parameter:

When you run sugar-emulator or sugar-runner a window opens up and the Sugar environment starts up and runs inside it.Â  It looks like this:

We can't just open the file with one click in the Journal because our program did not create the Journal entry and there are several Activities that support the MIME type of the Journal entry.Â  We need to use the Start With menu option like this:

When we do open the Journal entry this is what we see:

Technically, this is the first iteration of our Activity.Â  (Iteration is a vastly useful word meaning something you do more than once.Â  In this book we're building our Activity a bit at a time so I can demonstrate Activity writing principles, but actually building a program in pieces, testing it, getting feedback, and building a bit more can be a highly productive way of creating software.Â  Using the word iteration to describe each step in the process makes the process sound more formal than it really is).

While this Activity might be good enough to show your own mother, we really should improve it a bit before we do that.Â  That part comes next.

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## ADD REFINEMENTS

# Add Refinements

## Toolbars

It is a truth universally acknowledged that a first rate Activity needs good Toolbars.Â  In this chapter we'll learn how to make them.Â  We put the toolbar classes in a separate file from the rest. Originally this was because there used to be two styles of toolbar (old and new) and I wanted to support both in the Activity. With Sugar 3 there is only one style of toolbar so I could have put everything in one file.

The toolbar code is in a file called toolbar.py in the Add_Refinements_gtk directory of the Git repository. It looks like this:

Another file in the same directory of the Git repository is named ReadEtextsActivity2.py. Â  It looks like this:

This is the activity.info for this example:

When we run this new version this is what we'll see:

There are a few things worth pointing out in this code.Â  First, have a look at this import:

We'll be using the gettext module of Python to support translating our Activity into other languages. We'll be using it in statements like this one:

The underscore acts the same way as the gettext function because of the way we imported gettext.Â  The effect of this statement will be to look in a special translation file for a word or phrase that matches the key "Back" and replace it with its translation.Â  If there is no translation file for the language we want then it will simply use the word "Back".Â  We'll explore setting up these translation files later, but for now using gettext for all of the words and phrases we will show to our Activity users lays some important groundwork.

The second thing worth pointing out is that while our revised Activity has three toolbars we only had to create one of them.Â  The other two, Activity and Edit , are part of the Sugar Python library.Â  We can use those toolbars as is, hide the controls we don't need, or even extend them by adding new controls.Â  In the example we're hiding theÂ Share control of the Activity toolbar and the Undo , Redo , and Paste buttons of the Edit toolbar.Â  We currently do not support sharing books or modifying the text in books so these controls are not needed.

Another thing to notice is that the Activity class doesn't just provide us with a window.Â  The window has a VBox to hold our toolbar box and the body of our Activity.Â  We install the toolbox using set_toolbar_box() and the body of the Activity using set_canvas() .

The Read and View toolbars are regular PyGtk programming, but notice that there is a special button for Sugar toolbars that can have a tooltip attached to it, plus the View toolbar has code to hide the toolbox and ReadEtextsActivity2 has code to unhide it.Â  This is an easy function to add to your own Activities and many games and other kinds of Activities can benefit from the increased screen area you get when you hide the toolbox.

## Metadata And Journal Entries

Every Journal entry represents a single file plus metadata , or information describing the file.Â  There are standard metadata entries that all Journal entries have and you can also create your own custom metadata.

Unlike ReadEtextsActivity, this version has a write_file() method.

We didn't have a write_file() method before because we weren't going to update the file the book is in, and we still aren't.Â  We will, however, be updating the metadata for the Journal entry.Â  Specifically, we'll be doing two things:

- Save the page number our Activity user stopped reading on so when he launches the Activity again we can return to that page.
- Tell the Journal entry that it belongs to our Activity, so that in the future it will use our Activity's icon and can launch our Activity with one click.
The way the Read Activity saves page number is to use a custom metadata property.Â

Read creates a custom metadata property named Read_current_page to store the current page number.Â  You can create any number of custom metadata properties just this easily, so you may wonder why we aren't doing that with Read Etexts .Â  Actually, the first version of Read Etexts did use a custom property, but in Sugar .82 or lower there was a bug in the Journal such that custom metadata did not survive after the computer was turned off.Â  As a result my Activity would remember pages numbers while the computer was running, but would forget them as soon as it was shut down. This has not been a problem with Sugar for quite some time, but it was a real problem when the first edition of the book came out.

You might find how I got around this problem instructive. I created the following two methods:

save_page_number() looks at the current title metadata and either adds a page number to the end of it or updates the page number already there.Â  Since title is standard metadata for all Journal entries the Journal bug does not affect it.

These examples show how to read metadata too. Â

This line of code says "Get the metadata property named title and put it in the variable named title , If there is no title property put an empty string in title .

GenerallyÂ  you will save metadata in the write_file() method and read it in the read_file() method.

In a normal Activity that writes out a file in write_file() this next line would be unnecessary:

Any Journal entry created by an Activity will automatically have this property set. In the case of Pride and Prejudice , our Activity did not create it.Â  We are able to read it because our Activity supports its MIME type .Â  Unfortunately, that MIME type, application/zip , is used by other Activities.Â  I found it very frustrating to want to open a book in Read Etexts and accidentally have it opened in EToys instead.Â  This line of code solves that problem.Â  You only need to use Start Using... the first time you read a book.Â  After that the book will use the Read Etexts icon and can be resumed with a single click.

This does not at all affect the MIME type of the Journal entry, so if you wanted to deliberately open Pride and Prejudice with Etoys it is still possible.

Before we leave the subject of Journal metadata let's look at all the standard metadata that every Activity has.Â  Here is some code that creates a new Journal entry and updates a bunch of standard properties:

This code is taken from an Activity I wrote that downloads books from a website and creates Journal entries for them.Â  The Journal entries contain a friendly title and a full description of the book.

Most Activities will only deal with one Journal entry by using the read_file() and write_file() methods but you are not limited to that.Â  In a later chapter I'll show you how to create and delete Journal entries, how to list the contents of the Journal, and more.

- Putting your Activity in version control.Â  This will enable you to share your code with the world and get other people to help work on it.
- Getting your Activity translated into other languages.
- Distributing your finished Activity.Â  (Or your not quite finished but still useful Activity).
- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## ADD YOUR ACTIVITY CODE TO VERSION CONTROL

# Add Your Activity Code To Version Control

## What Is Version Control?

"If I have seen further it is only by standing on the shoulders of giants."

Isaac Newton, in a letter to Robert Hooke.

Writing an Activity is usually not something you do by yourself.Â  You will usually have collaborators in one form or another.Â  When I started writing Read Etexts I copied much of the code from the Read Activity.Â  When I implemented text to speech I adapted a toolbar from the Speak Activity.Â  When I finally got my copied file sharing code working the author of Image Viewer thought it was good enough to copy into that Activity.Â  Another programmer saw the work I did for text to speech and thought he could do it better.Â  He was right, and his improvements got merged into my own code.Â  When I wrote Get Internet Archive Books someone else took the user interface I came up with and made a more powerful and versatile Activity called Get Books .Â  Like Newton, everyone benefits from the work others have done before.

Even if I wanted to write Activities without help I would still need collaborators to translate them into other languages.

To make collaboration possible you need to have a place where everyone can post their code and share it.Â  This is called a code repository.Â  It isn't enough to just share the latest version of your code.Â  What you really want to do is share every version of your code.Â  Every time you make a significant change to your code you want to have the new version and the previous version available.Â  Not only do you want to have every version of your code available, you want to be able to compare any two versions your code to see what changed between them.Â  This is what version control software does.

The three most popular version control tools are CVS , Subversion , and Git .Â  Git is the newest and is the one used by Sugar Labs.Â  While not every Activity has its code in the Sugar Labs Git repository (other free code repositories exist) there is no good reason not to do it and significant benefits if you do.Â  If you want to get your Activity translated into other languages using the Sugar Labs Git repository is a must. Â

## Git Along Little Dogies

Git is a distributed version control system.Â  This means that not only are there copies of every version of your code in a central repository, the same copies exist on every user's computer.Â  This means you can update your local repository while you are not connected to the Internet, then connect and share everything at one time.

There are two ways you will interact with your Git repository: through Git commands and through the website at http://git.sugarlabs.org/. Â Â  We'll look at this website first.

Go to http://git.sugarlabs.org/ Â  and click on the Projects link in the upper right corner:

You will see a list of projects in the repository.Â  They will be listed from newest to oldest.Â  You'll also see a New Project link but you'll need to create an account to use that and we aren't ready to do that yet.

If you use the Search link in the upper right corner of the page you'll get a search form.Â  Use it to search for "read etexts".Â  Click on the link for that project when you find it.Â  You should see something like this:

This page lists some of the activity for the project but I don't find it particularly useful.Â  To get a much better look at your project start by clicking on the repository name on the right side of the page.Â  In this case the repository is named mainline .

You'll see something like this at the top of the page:

This page has some useful information on it.Â  First, have a look at the Public clone url and the HTTP clone url .Â  You need to click on More info... to see either one.Â  If you run either of these commands from the console you will get a copy of the git repository for the project copied to your computer.Â  This copy will include every version of every piece of code in the project.Â  You would need to modify it a bit before you could share your changes back to the main repository, but everything would be there.

The list under Activities is not that useful, but if you click on the Source Tree link you'll see something really good:

Here is a list of every file in the project, the date it was last updated, and a comment on what was modified.Â  Click on the link for ReadEtextsActivity.py and you'll see this:

This is the latest code in that file in pretty print format.Â  Python keywords are shown in a different color, there are line numbers, etc.Â  This is a good page for looking at code on the screen, but it doesn't print well and it's not much good for copying snippets of code into Eric windows either.Â  For either of those things you'll want to click on raw blob data at the top of the listing:

We're not done yet.Â  Use the Back button to get back to the pretty print listing and click on the Commits link.Â  This will give us a list of everything that changed each time we committed code into Git:

You may have noticed the odd combination of letters and numbers after the words James Simmons committed .Â  This is a kind of version number.Â  The usual practice with version control systems is to give each version of code you check in a version number, usually a simple sequence number.Â  Git is distributed, with many separate copies of the repository being modified independently and then merged.Â  That makes using just a sequential number to identify versions unworkable.Â  Instead, Git gives each version a really, really large random number.Â  The number is expressed in base 16, which uses the symbols 0-9 and a-f.Â  What you see in green is only a small part of the complete number.Â  The number is a link, and if you click on it you'll see this:

At the top of the page we see the complete version number used for this commit.Â  Below the gray box we see the full comment that was used to commit the changes.Â  Below that is a listing of what files were changed.Â Â  If we look further down the page we see this:

This is a diff report which shows the lines that have changed between this version and the previous version.Â  For each change it shows a few lines before and after the change to give you a better idea of what the change does.Â  Every change shows line numbers too.

A report like this is a wonderful aid to programming.Â  Sometimes when you're working on an enhancement to your program something that had been working mysteriously stops working.Â  When that happens you will wonder just what you changed that could have caused the problem.Â  A diff report can help you find the source of the problem.

By now you must be convinced that you want your project code in Git.Â  Before we can do that we need to create an account on this website.Â  That is no more difficult than creating an account on any other website, but it will need an important piece of information from us that we don't have yet.Â  Getting that information is our next task.

## Setting Up SSH Keys

To send your code to the Gitorious code repository you need an SSH public/private key pair.Â â SSH is a way of sending data over the network in encrypted format.Â  (In other words, it uses a secret code so nobody but the person getting the data can read it).Â  Public/private key encryption is a way of encrypting data that provides a way to guarantee that the person who is sending you the data is who he claims to be.

In simple terms it works like this: the SSH software generates two very large numbers that are used to encode and decode the data going over the network.Â  The first number, called the private key , is kept secret and is only used by you to encode the data.Â  The second number, called the public key , is given to anyone who needs to decode your data.Â  He can decode it using the public key; there is no need for him to know the private key.Â  He can also use the public key to encode a message to send back to you and you can decode it using your private key.

Git uses SSH like an electronic signature to verify that code changes that are supposed to be coming from you actually are coming from you.Â  The Git repository is given your public key.Â  It knows that anything it decodes with that key must have been sent by you because only you have the private key needed to encode it.

We will be using a tool called OpenSSH to generate the public and private keys.Â  This is included with every version of Linux so you just need to verify that it has been installed.Â  Then use the ssh-keygen utility that comes with OpenSSH to generate the keys:

By default ssh-keygen generates an RSA key, which is the kind we want.Â  By default it puts the keyfiles in a directory called / yourhome /.ssh and we want that too, so DO NOT enter a filename when it asks you to.Â  Just hit the Enter key to continue.

Now we DO want a passphrase here.Â  A passphrase is like a password that is used with the public and private keys to do the encrypting.Â  When you type it in you will not be able to see what you typed.Â  Because of that it will ask you to type the same thing again, and it will check to see that you typed them in the same way both times.

When choosing a passphrase remember that it needs to be something you can type reliably without seeing it and it would be better if it was not a word you can find in the dictionary, because those are easily broken. When I need to make a password I use the tool at http://www.multicians.org/thvv/gpw.html. Â  This tool generates a bunch of nonsense words that are pronounceable.Â  Pick one that appeals to you and use that.

Now have a look inside the .ssh directory.Â  By convention every file or directory name that begins with a period is considered hidden by Linux, so it won't show up in a GNOME file browser window unless you use the option on the View menu to Show Hidden Files.Â  When you display the contents of that directory you'll see two files: id_rsa and id_rsa.pub .Â  The public key is in id_rsa.pub.Â  Try opening that file with gedit (Open With Text Editor) and you'll see something like this:

When you create your account on git.sugarlabs.org there will be a place where you can add your public SSH key.Â  To do that use Select All from the Edit menu in gedit, then Copy and Paste into the field provided on the web form.

## Create A New Project

I'm going to create a new Project in Git for the examples for this book.Â  I need to log in with my new account and click the New Project link we saw earlier.Â  I get this form, which I have started filling in:

The Title is used on the website, the Slug is a shortened version of the title without spaces used to name the Git repository.Â Categories are optional.Â License is GPL v2 for my projects.Â  You can choose from any of the licenses in the list for your own Projects, and you can change the license entry later if you want to.Â  You will also need to enter a Description for your project.

Once you have this set up you'll be able to click on the mainline entry for the Project (like we did with Read Etexts before) and see something like this:

The next step is to convert our project files into a local Git repository, add the files to it, then push it to the repository on git.sugarlabs.org. Â  We need to do this because you cannot clone an empty repository, and our remote repository is currently empty.Â  To get around that problem we'll push the local repository out to the new remote repository we just created, then clone the remote one and delete our existing project and its Git repository.Â  From then on we'll do all our work in the cloned repository.

This process may remind you of the Edward Albee quote, " Sometimes a person has to go a very long distance out of his way to come back a short distance correctly". Fortunately we only need to do it once per project.Â  Enter the commands shown below in bold after making you project directory the current one:

I have made an empty local Git repository with git init , then I've used git add to add the important files to it.Â  (In fact git add doesn't actually add anything itself; it just tells Git to add the file on the next git commit ).Â  Finally git commit with the options shown will actually put the latest version of these files in my new local repository.

To push this local repository to git.sugarlabs.org Â  we use the commands from the web page:

The lines in bold are the commands to enter, and everything else is messages that Git sends to the console.Â  I've split some of the longer Git commands with the backslash (\) to make them fit better on the printed page, and wrapped some output lines that would normally print on one line for the same reason.Â  It probably isn't clear what we're doing here and why, so let's take it step by step:

- The first command git remote add origin tells the remote Git repository that we are going to send it stuff from our local repository.
- The second command git push origin master actually sends your local Git repository to the remote one and its contents will be copied in.Â  When you enter this command you will be asked to enter the SSH pass phrase you created in the last section.Â  GNOME will remember this phrase for you and enter it for every Git command afterwards so you don't need to.Â  It will keep doing this until you log out or turn off the computer.
- The next step is to delete our existing files and our local Git repository (which is contained in the hidden directory .git).Â  The rm .git -rf means "Delete the directory .git and everything in it".Â rm is a Unix command, not part of Git.Â  If you like you can delete your existing files after you create the cloned repository in the next step.Â  Note the command rm Activity/ReadEtextsII , which deletes the symbolic link to our old project that we created by running ./setup.py dev .Â  We'll need to go to our new cloned project directory and run that again before we can test our Activity again.
- Now we do the git clone command from the web page.Â  This takes the remote Git repository we just added our MANIFEST file to and makes a new local repository in directory / yourhome /olpc/bookexamples/mainline.
Finally we have a local repository we can use.Â  Well, not quite.Â  We can commit our code to it but we cannot push anything back to the remote repository because our local repository isn't configured correctly yet.

What we need to do is edit the file config in directory .git in / yourhome /olpc/bookexamples/mainline.Â We can use gedit to do that.Â  We need to change the url= entry to point to the Push url shown on the mainline web page.Â  When we're done our config file should look like this:

The line in bold is the only one that gets changed.Â  It is split here to make it fit on the printed page.Â  In your own files it should all be one line with no spaces between the colon(:) that ends the first line and the beginning of the second line.

From now on anyone who wants to work on our project can get a local copy of the Git repository by doing this from within the directory where he wants the repository to go:

He'll have to change his .git/config file just like we did, then he'll be ready to go.

## Everyday Use Of Git

While getting the repositories set up to begin with is a chore, daily use is not.Â  There are only a few commands you'll need to work with.Â  When we left off we had a repository in / yourhome /olpc/bookexamples/mainline with our files in it.Â  We will need to add any new files we create too.

We use the git add command to tell Git that we want to use Git to store a particular file.Â  This doesn't actually store anything, it just tells Git our intentions.Â  The format of the command is simply:

There are files we don't want to add to Git, to begin with those files that end in .pyc .Â  If we never do a git add on them they'll never get added, but Git will constantly ask us why we aren't adding them.Â  Fortunately there is a way to tell Git that we really, really don't want to add those files.Â  We need to create a file named .gitignore using gedit and put in entries like this:

These entries will also ignore project files used by Eric and zip files containing ebooks, The dist entry refers to the dist directory, which is where packaged Activities go, and the locale entry refers to a locale directory that will be created when we package the Activity. Neither of these directories belong in the Git repository. Once we have the .gitignore file created in the mainline directory we can add it to the repository:

From now on Git will no longer ask us to add .pyc or other unwantedÂ  files that match our patterns. If there are other files we don't want in the repository we can add them to .gitignore either as full file names or directory names or as patterns like *.pyc.

Â In addition to adding files to Git we can remove them too:

Note that this just tells Git that from now on it will not be keeping track of a given filename, and that will take effect at the next commit.Â  Old versions of the file are still in the repository.

If you want to see whatÂ changes will be applied at the next commit run this:

Finally, to put your latest changes in the repository use this:

If you leave off the -m an editor will open up and you can type in a comment, then save and exit. Unfortunately by default the editor is vi , an old text mode editor that is not friendly like gedit.

When we have all our changes done we can send them to the central repository using git push :

We can get the latest changes from other developers by doing git pull :

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## GOING INTERNATIONAL WITH POOTLE

# Going International With Pootle

## Introduction

The goal of Sugar Labs and One Laptop Per Child is to educate all the children of the world, and we can't do that with Activities that are only available in one language.Â  It is equally true that making separate versions of each Activity for every language is not going to work, and expecting Activity developers to be fluent in many languages is not realistic either.Â  We need a way for Activity developers to be able to concentrate on creating Activities and for those who can translate to just do that.Â  Fortunately, this is possible and the way it's done is by using gettext .

## Getting Text With gettext

You should remember that our latest code example made use of an odd import:

The "_()" function was used in statements like this:

At the time I explained that this odd looking function was used to translate the word "Back" into other languages, so that when someone looks at the Back button's tool tip he'll see the text in his own language.Â  I also said that if it was not possible to translate this text the user would see the word "Back" untranslated.Â  In this chapter we'll learn more about how this works and what we have to do to support the volunteers who translate these text strings into other languages.

The first thing you need to learn is how to properly format the text strings to be translated.Â  This is an issue when the text strings are actual sentences containing information.Â  For example, you might write such a message this way:

This would work, but you've made things difficult for the translator. Â  He has two separate strings to translate and no clue that they belong together.Â  It is much better to do this:

If you know both statements give the same resulting string then you can easily see why a translator would prefer the second one.Â  Use this technique whenever you need a message that has some information inserted into it.Â  When you use it, try and limit yourself to only one format code (the %s) per string.Â  If you use more than one it can cause problems for the translator.

## Going To Pot

Assuming that every string of text a user might be shown by our Activity is passed through "_()" the next step is to generate a pot file.Â  You can do this by running setup.py with a special option:

This creates a directory called po and puts a file ActivityName .pot in that directory.Â  In the case of our example project ActivityName is ReadEtextsII .Â  This is the contents of that file:

This file contains an entry for every text string in our Activity (as msgid) and a place to put a translation of that string (msgstr).Â Â  Copies of this file will be made by the Pootle server for every language desired, and the msgstr entries will be filled in by volunteer translators.

## Going To Pootle

Before any of that can happen we need to get our POT file into Pootle.Â  The first thing we need to do is get the new directory into our Git repository and push it out to Gitorious.Â  You should be familiar with the needed commands by now:

Next we need to give the user "pootle" commit authority to our Git project.Â  Go to git.sugarlabs.org, Â  sign in, and find your Project page and click on the mainline link.Â  You should see this on the page that takes you to:

Â

Click on the Add committer link and type in the name pootle in the form that takes you to.Â  When you come back to this page pootle will be listed under Committers.

Your next step is to go to web site http://bugs.sugarlabs.org Â  and register for a user id.Â  When you get that open up a ticket something like this:

The Component entry localization should be used, along with Type task .

Believe it or not, this is all you need to do to get your Activity set up to be translated.

## Pay No Attention To That Man Behind The Curtain

After this you'll need to do a few things to get translations from Pootle into your Activity.

- When you add text strings (labels, error messages, etc.) to your Activity always use the _() function with them so they can be translated.
- After adding new strings always run ./setup.py genpot to recreate the POT file.
- After that commit and push your changes to Gitorious.
- Every so often, and especially before releasing a new version, do a git pull .Â  If there are any localization files added to Gitorious this will bring them to you.
Localization with Pootle will create a large number of files in your project, some in the po directory and others in a new directory called locale .Â  These will be included in the .xo file that you will use to distribute your Activity.Â  The locale directory does not belong in the Git repository, however.Â  It will be generated fresh each time you package your Activity.

## C'est Magnifique!

Here is a screen shot of the French language version of Read Etexts reading Jules Verne's novel Vingt Mille Lieues Sour Le Mer :

Â

There is reason to believe that the book is in French too.

If you want to see how your own Activity looks in different languages (and you have localization files from Pootle for your Activity) all you need to do is bring up the Settings menu option in Sugar (shown here in French):

Then choose the Language icon like this:

And finally, pick your language. You will be prompted to restart Sugar. After you do you'll see everything in your new chosen language.

And that's all there is to it!

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## DISTRIBUTE YOUR ACTIVITY

# Distribute Your Activity

## Choose A License

Before you give your Activity to anyone you need to choose a license that it will be distributed under.Â  Buying software is like buying a book.Â  There are certain rights you have with a book and others you don't have.Â  If you buy a copy of The DaVinci Code you have the right to read it, to loan it out, to sell it to a used bookstore, or to burn it.Â  You do not have the right to make copies of it or to make a movie out of it.Â  Software is the same way, but often worse.Â  Those long license agreements we routinely accept by clicking a button might not allow you to sell the software when you're done with it, or even give it away.Â  If you sell your computer you may find that the software you bought is only good for that computer, and only while you are the owner of the computer.Â  (You can get good deals on reconditioned computers with no operating system installed for that very reason).

If you are in the business of selling software you might have to hire a lawyer to draw up a license agreement, but if you're giving away software there are several standard licenses you can choose from for free.Â  The most popular by far is called the General Public License , or GPL.Â  Like the licenses Microsoft uses it allows the people who get your program to do some things with it but not others.Â  What makes it interesting is not what it allows them to do (which is pretty much anything they like) but what it forbids them to do.

If someone distributes a program licensed under the GPL they are also required to make the source code of the program available to anyone who wants it.Â  That person may do as he likes with the code, with one important restriction: if he distributes a program based on that code he must also license that code using the GPL.Â  This makes it impossible for someone to take a GPL licensed work, improve it, and sell it to someone without giving him the source code to the new version.

While the GPL is not the only license available for Activities to be distributed on http://activities.sugarlabs.org Â  all the licenses require that anyone getting the Activity also gets the complete source code for it.Â  You've already taken care of that requirement by putting your source code in Gitorious.Â  If you used any code from an existing Activity licensed with the GPL you must license your own code the same way.Â  If you used a significant amount of code from this book (which is also GPL licensed) you may be required to use the GPL too.

Is licensing something you should worry about?Â  Not really.Â  The only reason you'd want to use a license other than the GPL is if you wanted to sell your Activity instead of give it away.Â  Consider what you'd have to do to make that possible:

- You'd have to use some language other than Python so you could give someone the program without giving them the source code.
- You would have to have your own source code repository not available to the general public and make arrangements to have the data backed up regularly.
- You would have to have your own website to distribute the Activity.Â  The website would have to be set up to accept payments somehow.
- You would have to advertise this website somehow or nobody would know your Activity existed.
- You would have to have a lawyer draw up a license for your Activity.
- You would have to come up with some mechanism to keep your customers from giving away copies of your Activity.
- You would have to create an Activity so astoundingly clever that nobody else could make something similar and give it away.
- You would have to deal with the fact that your "customers" would be children with no money or credit cards.
In summary, activities.sugarlabs.org Â  is not the iPhone App Store .Â  It is a place where programmers share and build upon each other's work and give the results to children for free.Â  The GPL encourages that to happen, and I recommend that you choose that for your license.

## Add License Comments To Your Python Code

At the top of each Python source file in your project (except setup.py , which is already commented) put comments like this:

Â

If the code is based on someone else's code you should mention that as a courtesy.

## Create An .xo File

Make certain that activity.info has the version number you want to give your Activity (currently it must be a positive integer) and run this command:

This will create a dist directory if one does not exist and put a file named something like ReadETextsII-1.xo in it.Â  The "1" indicates version 1 of the Activity.

If you did everything right this .xo file should be ready to distribute.Â  You can copy it to a thumb drive and install it on an XO laptop or onto another thumb drive running Sugar on a Stick .Â  You probably should do that before distributing it any further.Â  I like to live with new versions of my Activities for a week or so before putting them on activities.sugarlabs.org.

Now would be a good time to add dist and locale to your .gitignore file (if you haven't done that already), then commit it and push it to Gitorious.Â  You don't want to have copies of your .xo files in Git.Â  Another good thing to do at this point would be to tag your Git repository with the version number so you can identify which code goes with which version.

## Add Your Activity To ASLO

When you're ready to post the .xo file on ASLO you'll create an account as you did with the other websites.Â  When you've logged in there you'll see a Tools link in the upper right corner of the page. Click on that and you'll see a popup menu with an option for Developer Hub , which you should click on.Â  That will take you to the pages where you can add new Activities.Â  The first thing it asks for when setting up a new Activity is what license you will use.Â  After that you should have no problem getting your Activity set up.

You will need to create an Activity icon as a .gif file and create screen shots of your Activity in action.Â  You can do both of these things with The GIMP (GNU Image Manipulation Program).Â  For the icon all you need to do is open the .svg file with The GIMP and Save As a .gif file.

For the screen shots use sugar-emulator to display your Activity in action, then use the Screenshot option from the Create submenu of the File menu with these options:

This tells GIMP to wait 10 seconds, then take a screenshot of the window you click on with the mouse.Â  You'll know that the 10 seconds are up because the mouse pointer will change shape to a plus (+) sign.Â  You also tell it not to include the window decoration (which means the window title bar and border).Â  Since windows in Sugar do not have decorations eliminating the decorations used by sugar-emulator will give you a screenshot that looks exactly like a Sugar Activity in action.

Every Activity needs one screenshot, but you can have more if you like.Â  Screenshots help sell the Activity and instruct those who will use it on what the Activity can do.Â  Unfortunately, ASLO cannot display pictures in a predictable sequence, so it is not suited to displaying steps to perform.

Another thing you'll need to provide is a home page for your Activity.Â  The one for Read Etexts is here:

http://wiki.sugarlabs.org/go/Activities/Read_Etexts

Yes, one more website to get an account for.Â  Once you do you can specify a link with /go/Activities/some_name and when you click on that link the Wiki will create a page for you.Â  The software used for the Wiki is MediaWiki , the same as used for Wikipedia .Â  Your page does not need to be as elaborate as mine is, but you definitely should provide a link to your source code in Gitorious.

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS

## DEBUGGING SUGAR ACTIVITIES

# Debugging Sugar Activities

## Introduction

No matter how careful you are it is reasonably likely that your Activity will not work perfectly the first time you try it out.Â  Debugging a Sugar Activity is a bit different than debugging a standalone program.Â  When you test a standalone program you just run the program itself.Â  If there are syntax errors in the code you'll see the error messages on the console right away, and if you're running under the Eric IDE the offending line of code will be selected in the editor so you can correct it and keep going.

With Sugar it's a bit different.Â  It's the Sugar environment, not Eric, that runs your program.Â  If there are syntax errors in your code you won't see them right away.Â  Instead, the blinking Activity icon you see when your Activity starts up will just keep on blinking for several minutes and then will just go away, and your Activity won't start up.Â  The only way you'll see the error that caused the problem will be to use the Log Activity .Â  If your program has no syntax errors but does have logic errors you won't be able to step through your code with a debugger to find them.Â  Instead, you'll need to use some kind of logging to trace through what's happening in your code, and again use the Log Activity to view the trace messages.Â  Now would be a good time to repeat some advice I gave before:

## Make A Standalone Version Of Your Program First

Whatever your Activity does, it's a good bet that 80% of it could be done by a standalone program which would be much less tedious to debug.Â  If you can think of a way to make your Activity runnable as either an Activity or a standalone Python program then by all means do it.

## Use PyLint, PyChecker, or PyFlakes

One of the advantages of a compiled language like C over an interpreted language like Python is that the compiler does a complete syntax check of the code before converting it to machine language.Â  If there are syntax errors the compiler gives you informative error messages and stops the compile.Â  There is a utility call lint which C programmers can use to do even more thorough checks than the compiler would do and find questionable things going on in the code.

Python does not have a compiler but it does have several lint-like utilities you can run on your code before you test it.Â  These utilities are pyflakes , pychecker , and pylint .Â  Any Linux distribution should have all three available.

### PyFlakes

Here is an example of using PyFlakes:

PyFlakes seems to do the least checking of the three, but it does find errors like these above that a human eye would miss.

### PyChecker

Here is PyChecker in action:

PyChecker not only checks your code, it checks the code you import, including Sugar code.

### PyLint

Here is PyLint, the most thorough of the three:

PyLint is the toughest on your code and your ego.Â  It not only tells you about syntax errors, it tells you everything someone might find fault with in your code.Â  This includes style issues that won't affect how your code runs but will affect how readable it is to other programmers.

## The Log Activity

When you start testing your Activities the Log Activity will be like your second home. It displays a list of log files in the left pane and when you select one it will display the contents of the file in the right pane.Â  Every time you run your Activity a new log file is created for it, so you can compare the log you got this time with what you got on previous runs.Â  The Edit toolbar is especially useful.Â  It contains a button to show the log file with lines wrapped (which is not turned on by default but probably should be).Â  It has another button to copy selections from the log to the clipboard, which will be handy if you want to show log messages to other developers.

The Tools toolbar has a button to delete log files.Â  I've never found a reason to use it.Â  Log files go away on their own when you shut down sugar-emulator.

Â

Here is what the Log Activity looks like showing a syntax error in your code:Â

Â

## Logging

Without a doubt the oldest debugging technique there is would be the simple print statement.Â  If you have a running program that misbehaves because of logic errors and you can't step through the code in a debugger to figure out what's happening you might print statements in your code.Â  For instance, if you aren't sure that a method is ever getting executed you might put a statement like this as the first line of the method:

You can include data in your print statements too.Â  Suppose you need to know how many times a loop is run.Â  You could do this:

The output of these print statements can be seen in the Log Activity.Â  When you're finished debugging your program you would remove these statements.

An old programming book I read once made the case for leaving the statements in the finished program.Â  The authors felt that using these statements for debugging and them removing them is a bit like wearing a parachute when the plane is on the ground and taking it off when it's airborne.Â  If the program is out in the world and has problems you might well wish you had those statements in the code so you could help the user and yourself figure out what's going on.Â  On the other hand, print statements aren't free.Â  They do take time to run and they fill up the log files with junk.Â  What we need are print statements that you can turn on an off.

The way you can do this is with Python Standard Logging.Â  In the form used by most Activities it looks like this:

These statements would go in the __init__() method of your Activity.Â  Every time you want to do a print() statement you would do this instead:

Notice that there are two kinds of logging going on here: debug and error . Â  These are error levels.Â  Every statement has one, and they control which log statements are run and which are ignored.Â  There are several levels of error logging, from lowest severity to highest:

When you set the error level in your program to one of these values you get messages with that level and higher.Â  You can set the level in your program code like this:

You can also set the logging level outside your program code using an environment variable .Â  For instance, in Sugar .82 and lower you can start sugar-emulator like this:

You can do the same thing with .84 and later, but there is a more convenient way.Â  Edit the file ~/.sugar/debug and uncomment the line that sets the SUGAR_LOGGER_LEVEL.Â  Whatever value you have for SUGAR_LOGGER_LEVEL in ~/.sugar/debug will override the one set by the environment variable, so either change the setting in the file or use the environment variable, but don't do both.

## The Analyze Activity

Another Activity you may find yourself using at some point is Analyze .Â  This is more likely to be used to debug Sugar itself than to debug your Activity.Â  If, for instance, your collaboration test environment doesn't seem to be working this Activity might help you or someone else figure out why.

I don't have a lot to say about this Activity here, but you should be aware that it exists.

Â

- SUGAR ACTIVITIES
- INTRODUCTION
- WHAT IS SUGAR?
- WHAT IS A SUGAR ACTIVITY?
- WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?
- PROGRAMMING
- SETTING UP A DEVELOPMENT ENVIRONMENT
- CREATING YOUR FIRST ACTIVITY
- A STANDALONE PYTHON PROGRAM FOR READING ETEXTS
- INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY
- PACKAGE THE ACTIVITY
- ADD REFINEMENTS
- ADD YOUR ACTIVITY CODE TO VERSION CONTROL
- GOING INTERNATIONAL WITH POOTLE
- DISTRIBUTE YOUR ACTIVITY
- DEBUGGING SUGAR ACTIVITIES
- ADVANCED TOPICS
- MAKING SHARED ACTIVITIES
- ADDING TEXT TO SPEECH
- FUN WITH THE JOURNAL
- MAKING ACTIVITIES USING PYGAME
- GUEST CHAPTERS
- DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT
- PORTING YOUR ACTIVITIES TO GTK3
- CONVERTING HIPPO CANVAS TO GTK3
- MAKING CONTRIBUTIONS TO SUGAR
- MAKING ACTIVITIES THAT USE THE ACCELEROMETER
- APPENDIX
- ABOUT THE AUTHORS
- WHERE TO GO FROM HERE?
- CREDITS
