> **Note:** This document is automatically generated from publicly available FLOSS Manuals content.
> For the original and most up-to-date version, visit: https://archive.flossmanuals.net/make-your-own-sugar-activities/index.html

# Sugar Activities Guide (sourced from FLOSS Manuals)

## Introduction

"This book is a record of a pleasure trip. If it were a record of a solemn scientific expedition, it would have about it that gravity, that profundity, and that impressive incomprehensibility which are so proper to works of that kind, and withal so attractive."

From the Preface to The Innocents Abroad , by Mark Twain

The purpose of this book is to teach you what you need to know to write Activities for Sugar, the operating environment developed for the One Laptop Per Child project. This book does not assume that you know how to program a computer, although those who do will find useful information in it. My primary goal in writing it is to encourage non programmers, including children and their teachers, to create their own Sugar Activities. Because of this goal I will include some details that other books would leave out and leave out things that others would include. Impressive incomprehensibility will be kept to a minimum.

If you just want to learn how to write computer programs Sugar provides many Activities to help you do that: Etoys, Turtle Art, Scratch, and Pippy. None of these are really suitable for creating Activities so I won't cover them in this book, but they're a great way to learn about programming. If you decide after playing with these that you'd like to try writing an Activity after all you'll have a good foundation of knowledge to build on.

When you have done some programming then you'll know how satisfying it can be to use a program that you made yourself, one that does exactly what you want it to do. Creating a Sugar Activity takes that enjoyment to the next level. A useful Sugar Activity can be translated by volunteers into every language, be downloaded hundreds of times a week and used every day by students all over the world.

A book that teaches everything you need to know to write Activities would be really, really long and would duplicate material that is already available elsewhere. Because of this, I am going to write this as sort of a guided tour of Activity development. That means, for example, that I'll teach you what Python is and why it's important to learn it but I won't teach you the Python language itself. There are excellent tutorials on the Internet that will do that, and I'll refer you to those tutorials.

There is much sample code in this book, but there is no need for you to type it in to try it out. All of the code is in a Git repository that you can download to your own computer. If you've never used Git there is a chapter that explains what it is and how to use it.

I started writing Activities shortly after I received my XO laptop. When I started I didn't know any of the material that will be in this book. I had a hard time knowing where to begin. What I did have going for me though was a little less than 30 years as a professional programmer. As a result of that I think like a programmer. A good programmer can take a complex task and divide it up into manageable pieces. He can figure out how things must work, and from that figure out how they do work. He knows how to ask for help and where. If there is no obvious place to begin he can begin somewhere and eventually get where he needs to go.

Because I went through this process I think I can be a pretty good guide to writing Sugar Activities. Along the way I hope to also teach you how to think like a programmer does.

From time to time I may add chapters to this book. Sugar is a great application platform and this book can only begin to tell you what is possible.

In the first edition of this book I expressed the wish that future versions of the book would have guest chapters on more advanced topics written by other experienced Activity developers. That wish has been realized. Not only does the book have guest chapters, but some of the new content has been created by young developers from the Google Code-in of 2012, for which I was one of the mentors. Their bios are in the About The Authors chapter. Read them and prepare to be impressed!

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

http://www.archive.org/details/MakeYourOwnSugarActivities

The Amazon Kindle Store has exactly the same MOBI version as the Internet Archive does.

If you choose to read this book on a Kindle or a Nook be aware that these devices have narrow screens not well suited for displaying program listings. I suggest you refer to the FLOSS Manuals website to see what the code looks like properly formatted.

There is a Spanish version of the first edition of this book, Cómo Hacer Una Actividad Sugar, which is available in all the same places as this version.

SUGAR ACTIVITIES

INTRODUCTION

WHAT IS SUGAR?

WHAT IS A SUGAR ACTIVITY?

WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?

PROGRAMMING

SETTING UP A DEVELOPMENT ENVIRONMENT

CREATING YOUR FIRST ACTIVITY

A STANDALONE PYTHON PROGRAM FOR READING ETEXTS

INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY

PACKAGE THE ACTIVITY

ADD REFINEMENTS

ADD YOUR ACTIVITY CODE TO VERSION CONTROL

GOING INTERNATIONAL WITH POOTLE

DISTRIBUTE YOUR ACTIVITY

DEBUGGING SUGAR ACTIVITIES

ADVANCED TOPICS

MAKING SHARED ACTIVITIES

ADDING TEXT TO SPEECH

FUN WITH THE JOURNAL

MAKING ACTIVITIES USING PYGAME

GUEST CHAPTERS

DEVELOPING SUGAR ACTIVITIES USING HTML5 AND WEBKIT

PORTING YOUR ACTIVITIES TO GTK3

CONVERTING HIPPO CANVAS TO GTK3

MAKING CONTRIBUTIONS TO SUGAR

MAKING ACTIVITIES THAT USE THE ACCELEROMETER

APPENDIX

ABOUT THE AUTHORS

WHERE TO GO FROM HERE?

CREDITS