The data in this directory make up the NCDC archive of DSI 3260 (15-minute precipitation for primarily U.S. locations). For years 2011 and earlier data files are 
organized into individual subdirectories by state using a two-digit state code that is defined in either of the documentation files - dsi3260.doc (Word format)
or dsi3260.pdf (portable document file) located within the same directory as this readme file. Within each state specific sub-directory the data files 
reside archived and compressed  in a tar.z file. These files may be uncompressed/unarchived with a number of standard decompression software utilities 
such as gzip or 7-zip. Each file contains data for all available data for all stations within a state for a period defined by the file name. Files are named as
3260_ss_bbbb-eeee.tar.z where "ss" is the two-digit state code defined in the documentation (dsi3260.pdf or dsi3260.doc), "bbbb"
is the four digit beginning year of the data and "eeee" is the ending year of the data. 

Beginning with the year 2012 the data is organized in one directories per year containing individual monthly files of all stations. The directories are named by_monthYYYY 
where "YYYY" is the 4-digit data year. The file names are 3260mmmYYYY.dat where "mmm" is a 3-character abbreviation for the month (jan, feb, mar, etc.) and "YYYY" is the 
4-digit data year. For example, all stations that reported in April 2012 will be in the by_month2012 directory in the file 3260apr2012.dat.

Please refer to dsi3260.pdf or dsi3260.doc for interpretation of the data files. 

Data can also be accessed via ftp The directory and all contents are accessible via web browser:
http://www1.ncdc.noaa.gov/pub/data/3260/
Or via direct ftp:
- Ftp to ftp.ncdc.noaa.gov
- login as ftp or as anonymous
- password is your email address
- cd to /pub/data/3260
- get or mget files as needed--in ascii or bin mode (bin required for any non-ascii file)
- cd to individual directories by state to get data files

Point of Contact - 

Please contact NCDC at 828-271-4800 or email hpd.ncdc@noaa.gov for information or assistance.
