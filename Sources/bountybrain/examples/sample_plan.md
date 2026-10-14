# Recon-informed testing plan

6 asset(s) matched against the knowledge graph. Scores blend signature applicability, exploitation likelihood (EPSS), public exploit availability, and historical bounty economics.

## blog.acme.example
**Fingerprinted:** Apache, Elementor, WordPress
**CVEs found by recon:** CVE-2022-29455

| # | Weakness | Score | EPSS | Exploits | Median bounty | Signals |
|---|----------|-------|------|----------|---------------|---------|
| 1 | [CWE-79 Improper Neutralization of Input During Web ](https://cwe.mitre.org/data/definitions/79.html) | 0.909 | 0.973 | 154 | $1,250 | 437 sigs, 1184 writeups |
| 2 | [CWE-89 Improper Neutralization of Special Elements ](https://cwe.mitre.org/data/definitions/89.html) | 0.632 | 0.978 | 57 | $3,000 | 173 sigs, 230 writeups |
| 3 | [CWE-22 Improper Limitation of a Pathname to a Restr](https://cwe.mitre.org/data/definitions/22.html) | 0.509 | 1.000 | 306 | $5,000 | 90 sigs, 203 writeups |
| 4 | [CWE-94 Improper Control of Generation of Code ('Cod](https://cwe.mitre.org/data/definitions/94.html) | 0.477 | 0.983 | 49 | $5,600 | 36 sigs, 880 writeups |
| 5 | [CWE-434 Unrestricted Upload of File with Dangerous T](https://cwe.mitre.org/data/definitions/434.html) | 0.452 | 1.000 | 34 | $2,400 | 56 sigs, 105 writeups |

### What to try: CWE-79 Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser or an automated tool, an attacker follows all public links and actions on a web site. They record all the links, the forms, the resources accessed and all other potential entry-points for the web application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all links visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_
- **Explore** — [Send request to target webpage and analyze HTML] Using a browser or an automated tool, an adversary sends requests to a webpage and records the received HTML response. Adversaries then analyze the HTML to identify any known underlying JavaScript architectures. This can aid in mappiong publicly known vulnerabilities to the webpage and can also helpo the adversary guess application architecture and the inner workings of a system.
    - Record all "src" values inside script tags. These JavaScript files are compared to lists of files for known architectures. If there is a large match between the "src" values and architecture files, then it can be assumed that particular architecture is being used.
    - _source: [CAPEC-85: AJAX Footprinting](https://capec.mitre.org/data/definitions/85.html)_
- **Experiment** — [Probe identified potential entry points for XSS vulnerability] The attacker uses the entry points gathered in the "Explore" phase as a target list and injects various common script payloads to determine if an entry point actually represents a vulnerability and to characterize the extent to which the vulnerability can be exploited.
    - Use a list of XSS probe strings to inject script in parameters of known URLs. If possible, the probe strings contain a unique identifier.
    - Use a proxy tool to record results of manual input of XSS probes in known URLs.
    - Use a list of XSS probe strings to inject script into UI entry fields. If possible, the probe strings contain a unique identifier.
    - Use a list of XSS probe strings to inject script into resources accessed by the application. If possible, the probe strings contain a unique identifier.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_

**Detection templates to run:**
- [Apache OFBiz <=16.11.07 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-1943.yaml) (medium)
- [Beautiful Cookie Consent Banner < 2.10.2 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2023/CVE-2023-3388.yaml) (high)
- [WordPress UserPro 4.9.32 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2019/CVE-2019-14470.yaml) (medium)

**Prior disclosures:**
- [We Hacked Apple for 3 Months: Here’s What We Found](https://samcurry.net/hacking-apple/) — $288,500
- [How I Hacked Microsoft Teams and got $150,000 in Pwn2Own](https://speakerdeck.com/masatokinugawa/how-i-hacked-microsoft-teams-and-got-150000-dollars-in-pwn2own) — $150,000
- [How we made $120k bug bounty in a year with good automation](https://www.vidocsecurity.com/blog/2022-summary-how-we-made-120k-bug-bounty-in-a-year/) — $120,000

### What to try: CWE-89 Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
- **Explore** — [Survey application] The attacker first takes an inventory of the functionality exposed by the application.
    - Spider web sites for all available links
    - Sniff network communications with application using a utility such as WireShark.
    - _source: [CAPEC-66: SQL Injection](https://capec.mitre.org/data/definitions/66.html)_
- **Explore** — [Hypothesize SQL queries in application]Generated hypotheses regarding the SQL queries in an application. For example, the adversary may hypothesize that their input is passed directly into a query that looks like: "SELECT * FROM orders WHERE ordernum = _____"or"SELECT * FROM orders WHERE ordernum IN (_____)"or"SELECT * FROM orders WHERE ordernum in (_____) ORDER BY _____" Of course, there are many other possibilities.
    - Research types of SQL queries and determine which ones could be used at various places in an application.
    - _source: [CAPEC-7: Blind SQL Injection](https://capec.mitre.org/data/definitions/7.html)_
- **Explore** — [Determine how to inject information into the queries]Determine how to inject information into the queries from the previous step such that the injection does not impact their logic. For example, the following are possible injections for those queries: "5' OR 1=1; --"and"5) OR 1=1; --"and"ordernum DESC; --"
    - Add clauses to the SQL queries such that the query logic does not change.
    - Add delays to the SQL queries in case server does not provide clear error messages (e.g. WAITFOR DELAY '0:0:10' in SQL Server or BENCHMARK(1000000000,MD5(1) in MySQL). If these can be injected into the queries, then the length of time that the server takes to respond reveals whether the query is injectable or not.
    - _source: [CAPEC-7: Blind SQL Injection](https://capec.mitre.org/data/definitions/7.html)_
- **Experiment** — [Determine user-controllable input susceptible to injection] Determine the user-controllable input susceptible to injection. For each user-controllable input that the attacker suspects is vulnerable to SQL injection, attempt to inject characters that have special meaning in SQL (such as a single quote character, a double quote character, two hyphens, a parenthesis, etc.). The goal is to create a SQL query with an invalid syntax.
    - Use web browser to inject input through text fields or through HTTP GET parameters.
    - Use a web application debugging tool such as Tamper Data, TamperIE, WebScarab,etc. to modify HTTP POST parameters, hidden fields, non-freeform fields, etc.
    - Use network-level packet injection tools such as netcat to inject input
    - Use modified client (modified by reverse engineering) to inject input.
    - _source: [CAPEC-66: SQL Injection](https://capec.mitre.org/data/definitions/66.html)_
- **Experiment** — [Experiment with SQL Injection vulnerabilities] After determining that a given input is vulnerable to SQL Injection, hypothesize what the underlying query looks like. Iteratively try to add logic to the query to extract information from the database, or to modify or delete information in the database.
    - Use public resources such as "SQL Injection Cheat Sheet" at http://ferruh.mavituna.com/makale/sql-injection-cheatsheet/, and try different approaches for adding logic to SQL queries.
    - Add logic to query, and use detailed error messages from the server to debug the query. For example, if adding a single quote to a query causes an error message, try : "' OR 1=1; --", or something else that would syntactically complete a hypothesized query. Iteratively refine the query.
    - Use "Blind SQL Injection" techniques to extract information about the database schema.
    - If a denial of service attack is the goal, try stacking queries. This does not work on all platforms (most notably, it does not work on Oracle or MySQL). Examples of inputs to try include: "'; DROP TABLE SYSOBJECTS; --" and "'); DROP TABLE SYSOBJECTS; --". These particular queries will likely not work because the SYSOBJECTS table is generally protected.
    - _source: [CAPEC-66: SQL Injection](https://capec.mitre.org/data/definitions/66.html)_
- **Experiment** — [Determine user-controllable input susceptible to injection] Determine the user-controllable input susceptible to injection. For each user-controllable input that the adversary suspects is vulnerable to SQL injection, attempt to inject the values determined in the previous step. If an error does not occur, then the adversary knows that the SQL injection was successful.
    - Use web browser to inject input through text fields or through HTTP GET parameters.
    - Use a web application debugging tool such as Tamper Data, TamperIE, WebScarab,etc. to modify HTTP POST parameters, hidden fields, non-freeform fields, etc.
    - Use network-level packet injection tools such as netcat to inject input
    - Use modified client (modified by reverse engineering) to inject input.
    - _source: [CAPEC-7: Blind SQL Injection](https://capec.mitre.org/data/definitions/7.html)_

**Detection templates to run:**
- [WordPress <5.8.3 - SQL Injection](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2022/CVE-2022-21661.yaml) (high)
- [WordPress Chop Slider 3 - Blind SQL Injection](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-11530.yaml) (critical)
- [WordPress PayPal Pro <1.1.65 - SQL Injection](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-14092.yaml) (critical)

**Prior disclosures:**
- [We Hacked Apple for 3 Months: Here’s What We Found](https://samcurry.net/hacking-apple/) — $288,500
- [Subdomain Fuzzing worth 35k bounty!](https://medium.com/@HX007/subdomain-fuzzing-worth-35k-bounty-daebcb56d9bc) — $35,000
- [WordPress Transposh: Exploiting a Blind SQL Injection via XSS - RCE Security](https://www.rcesecurity.com/2022/07/WordPress-Transposh-Exploiting-a-Blind-SQL-Injection-via-XSS/) — $30,000

### What to try: CWE-22 Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser, an automated tool or by inspecting the application, an adversary records all entry points to the application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all user input entry points visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - Manually inspect the application to find entry points.
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Explore** — The attacker accesses the server using a specific URL.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_
- **Experiment** — [Probe entry points to locate vulnerabilities] The adversary uses the entry points gathered in the "Explore" phase as a target list and looks for areas where user input is used to access resources on the target host. The adversary attempts different encodings of slash characters to bypass input filters.
    - Try both backslash and forward slash characters
    - Try different encodings for slash characters such as %5C
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Experiment** — The attacker tries to encode some special characters in the URL. The attacker find out that some characters are not filtered properly.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_

**Detection templates to run:**
- [Apache 2.4.49 - Path Traversal and Remote Code Execution](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2021/CVE-2021-41773.yaml) (high)
- [Apache 2.4.49/2.4.50 - Path Traversal and Remote Code Execution](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2021/CVE-2021-42013.yaml) (critical)
- [Apache OFBiz Directory Traversal - Remote Code Execution](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2024/CVE-2024-32113.yaml) (high)

**Prior disclosures:**
- [GitHub Pages - Multiple RCEs via insecure Kramdown configuration - $25,000 Bount](https://devcraft.io/2020/10/20/github-pages-multiple-rces-via-kramdown-config.html) — $25,000
- [CS:GO: From Zero to 0-day](https://neodyme.io/blog/csgo_from_zero_to_0day/) — $22,500
- [Two weeks of securing Samsung devices: Part 2](https://blog.oversecured.com/Two-weeks-of-securing-Samsung-devices-Part-2/) — $18,040

## www.acme.example
**Fingerprinted:** PHP, WordPress, jQuery, nginx

| # | Weakness | Score | EPSS | Exploits | Median bounty | Signals |
|---|----------|-------|------|----------|---------------|---------|
| 1 | [CWE-79 Improper Neutralization of Input During Web ](https://cwe.mitre.org/data/definitions/79.html) | 0.890 | 0.845 | 154 | $1,250 | 432 sigs, 1184 writeups |
| 2 | [CWE-89 Improper Neutralization of Special Elements ](https://cwe.mitre.org/data/definitions/89.html) | 0.635 | 0.978 | 57 | $3,000 | 173 sigs, 230 writeups |
| 3 | [CWE-22 Improper Limitation of a Pathname to a Restr](https://cwe.mitre.org/data/definitions/22.html) | 0.502 | 0.978 | 306 | $5,000 | 86 sigs, 203 writeups |
| 4 | [CWE-94 Improper Control of Generation of Code ('Cod](https://cwe.mitre.org/data/definitions/94.html) | 0.471 | 1.000 | 49 | $5,600 | 29 sigs, 880 writeups |
| 5 | [CWE-434 Unrestricted Upload of File with Dangerous T](https://cwe.mitre.org/data/definitions/434.html) | 0.445 | 0.973 | 34 | $2,400 | 53 sigs, 105 writeups |

### What to try: CWE-79 Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser or an automated tool, an attacker follows all public links and actions on a web site. They record all the links, the forms, the resources accessed and all other potential entry-points for the web application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all links visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_
- **Explore** — [Send request to target webpage and analyze HTML] Using a browser or an automated tool, an adversary sends requests to a webpage and records the received HTML response. Adversaries then analyze the HTML to identify any known underlying JavaScript architectures. This can aid in mappiong publicly known vulnerabilities to the webpage and can also helpo the adversary guess application architecture and the inner workings of a system.
    - Record all "src" values inside script tags. These JavaScript files are compared to lists of files for known architectures. If there is a large match between the "src" values and architecture files, then it can be assumed that particular architecture is being used.
    - _source: [CAPEC-85: AJAX Footprinting](https://capec.mitre.org/data/definitions/85.html)_
- **Experiment** — [Probe identified potential entry points for XSS vulnerability] The attacker uses the entry points gathered in the "Explore" phase as a target list and injects various common script payloads to determine if an entry point actually represents a vulnerability and to characterize the extent to which the vulnerability can be exploited.
    - Use a list of XSS probe strings to inject script in parameters of known URLs. If possible, the probe strings contain a unique identifier.
    - Use a proxy tool to record results of manual input of XSS probes in known URLs.
    - Use a list of XSS probe strings to inject script into UI entry fields. If possible, the probe strings contain a unique identifier.
    - Use a list of XSS probe strings to inject script into resources accessed by the application. If possible, the probe strings contain a unique identifier.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_

**Detection templates to run:**
- [Beautiful Cookie Consent Banner < 2.10.2 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2023/CVE-2023-3388.yaml) (high)
- [WordPress UserPro 4.9.32 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2019/CVE-2019-14470.yaml) (medium)
- [WordPress Social Warfare <3.5.3 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2019/CVE-2019-9978.yaml) (medium)

**Prior disclosures:**
- [We Hacked Apple for 3 Months: Here’s What We Found](https://samcurry.net/hacking-apple/) — $288,500
- [How I Hacked Microsoft Teams and got $150,000 in Pwn2Own](https://speakerdeck.com/masatokinugawa/how-i-hacked-microsoft-teams-and-got-150000-dollars-in-pwn2own) — $150,000
- [How we made $120k bug bounty in a year with good automation](https://www.vidocsecurity.com/blog/2022-summary-how-we-made-120k-bug-bounty-in-a-year/) — $120,000

### What to try: CWE-89 Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
- **Explore** — [Survey application] The attacker first takes an inventory of the functionality exposed by the application.
    - Spider web sites for all available links
    - Sniff network communications with application using a utility such as WireShark.
    - _source: [CAPEC-66: SQL Injection](https://capec.mitre.org/data/definitions/66.html)_
- **Explore** — [Hypothesize SQL queries in application]Generated hypotheses regarding the SQL queries in an application. For example, the adversary may hypothesize that their input is passed directly into a query that looks like: "SELECT * FROM orders WHERE ordernum = _____"or"SELECT * FROM orders WHERE ordernum IN (_____)"or"SELECT * FROM orders WHERE ordernum in (_____) ORDER BY _____" Of course, there are many other possibilities.
    - Research types of SQL queries and determine which ones could be used at various places in an application.
    - _source: [CAPEC-7: Blind SQL Injection](https://capec.mitre.org/data/definitions/7.html)_
- **Explore** — [Determine how to inject information into the queries]Determine how to inject information into the queries from the previous step such that the injection does not impact their logic. For example, the following are possible injections for those queries: "5' OR 1=1; --"and"5) OR 1=1; --"and"ordernum DESC; --"
    - Add clauses to the SQL queries such that the query logic does not change.
    - Add delays to the SQL queries in case server does not provide clear error messages (e.g. WAITFOR DELAY '0:0:10' in SQL Server or BENCHMARK(1000000000,MD5(1) in MySQL). If these can be injected into the queries, then the length of time that the server takes to respond reveals whether the query is injectable or not.
    - _source: [CAPEC-7: Blind SQL Injection](https://capec.mitre.org/data/definitions/7.html)_
- **Experiment** — [Determine user-controllable input susceptible to injection] Determine the user-controllable input susceptible to injection. For each user-controllable input that the attacker suspects is vulnerable to SQL injection, attempt to inject characters that have special meaning in SQL (such as a single quote character, a double quote character, two hyphens, a parenthesis, etc.). The goal is to create a SQL query with an invalid syntax.
    - Use web browser to inject input through text fields or through HTTP GET parameters.
    - Use a web application debugging tool such as Tamper Data, TamperIE, WebScarab,etc. to modify HTTP POST parameters, hidden fields, non-freeform fields, etc.
    - Use network-level packet injection tools such as netcat to inject input
    - Use modified client (modified by reverse engineering) to inject input.
    - _source: [CAPEC-66: SQL Injection](https://capec.mitre.org/data/definitions/66.html)_
- **Experiment** — [Experiment with SQL Injection vulnerabilities] After determining that a given input is vulnerable to SQL Injection, hypothesize what the underlying query looks like. Iteratively try to add logic to the query to extract information from the database, or to modify or delete information in the database.
    - Use public resources such as "SQL Injection Cheat Sheet" at http://ferruh.mavituna.com/makale/sql-injection-cheatsheet/, and try different approaches for adding logic to SQL queries.
    - Add logic to query, and use detailed error messages from the server to debug the query. For example, if adding a single quote to a query causes an error message, try : "' OR 1=1; --", or something else that would syntactically complete a hypothesized query. Iteratively refine the query.
    - Use "Blind SQL Injection" techniques to extract information about the database schema.
    - If a denial of service attack is the goal, try stacking queries. This does not work on all platforms (most notably, it does not work on Oracle or MySQL). Examples of inputs to try include: "'; DROP TABLE SYSOBJECTS; --" and "'); DROP TABLE SYSOBJECTS; --". These particular queries will likely not work because the SYSOBJECTS table is generally protected.
    - _source: [CAPEC-66: SQL Injection](https://capec.mitre.org/data/definitions/66.html)_
- **Experiment** — [Determine user-controllable input susceptible to injection] Determine the user-controllable input susceptible to injection. For each user-controllable input that the adversary suspects is vulnerable to SQL injection, attempt to inject the values determined in the previous step. If an error does not occur, then the adversary knows that the SQL injection was successful.
    - Use web browser to inject input through text fields or through HTTP GET parameters.
    - Use a web application debugging tool such as Tamper Data, TamperIE, WebScarab,etc. to modify HTTP POST parameters, hidden fields, non-freeform fields, etc.
    - Use network-level packet injection tools such as netcat to inject input
    - Use modified client (modified by reverse engineering) to inject input.
    - _source: [CAPEC-7: Blind SQL Injection](https://capec.mitre.org/data/definitions/7.html)_

**Detection templates to run:**
- [WordPress <5.8.3 - SQL Injection](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2022/CVE-2022-21661.yaml) (high)
- [WordPress Chop Slider 3 - Blind SQL Injection](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-11530.yaml) (critical)
- [WordPress PayPal Pro <1.1.65 - SQL Injection](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-14092.yaml) (critical)

**Prior disclosures:**
- [We Hacked Apple for 3 Months: Here’s What We Found](https://samcurry.net/hacking-apple/) — $288,500
- [Subdomain Fuzzing worth 35k bounty!](https://medium.com/@HX007/subdomain-fuzzing-worth-35k-bounty-daebcb56d9bc) — $35,000
- [WordPress Transposh: Exploiting a Blind SQL Injection via XSS - RCE Security](https://www.rcesecurity.com/2022/07/WordPress-Transposh-Exploiting-a-Blind-SQL-Injection-via-XSS/) — $30,000

### What to try: CWE-22 Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser, an automated tool or by inspecting the application, an adversary records all entry points to the application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all user input entry points visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - Manually inspect the application to find entry points.
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Explore** — The attacker accesses the server using a specific URL.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_
- **Experiment** — [Probe entry points to locate vulnerabilities] The adversary uses the entry points gathered in the "Explore" phase as a target list and looks for areas where user input is used to access resources on the target host. The adversary attempts different encodings of slash characters to bypass input filters.
    - Try both backslash and forward slash characters
    - Try different encodings for slash characters such as %5C
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Experiment** — The attacker tries to encode some special characters in the URL. The attacker find out that some characters are not filtered properly.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_

**Detection templates to run:**
- [WordPress Duplicator 1.3.24 & 1.3.26 - Local File Inclusion](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-11738.yaml) (high)
- [WordPress File Upload <= 4.24.11 - Arbitrary File Read](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2024/CVE-2024-9047.yaml) (critical)
- [WordPress Core 5.0.0 - Crop-image Shell Upload](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2019/CVE-2019-8943.yaml) (medium)

**Prior disclosures:**
- [GitHub Pages - Multiple RCEs via insecure Kramdown configuration - $25,000 Bount](https://devcraft.io/2020/10/20/github-pages-multiple-rces-via-kramdown-config.html) — $25,000
- [CS:GO: From Zero to 0-day](https://neodyme.io/blog/csgo_from_zero_to_0day/) — $22,500
- [Two weeks of securing Samsung devices: Part 2](https://blog.oversecured.com/Two-weeks-of-securing-Samsung-devices-Part-2/) — $18,040

## jira.acme.example
**Fingerprinted:** Java, Jira, Tomcat

| # | Weakness | Score | EPSS | Exploits | Median bounty | Signals |
|---|----------|-------|------|----------|---------------|---------|
| 1 | [CWE-200 Exposure of Sensitive Information to an Unau](https://cwe.mitre.org/data/definitions/200.html) | 0.887 | 0.996 | 23 | $1,500 | 8 sigs, 672 writeups |
| 2 | [CWE-22 Improper Limitation of a Pathname to a Restr](https://cwe.mitre.org/data/definitions/22.html) | 0.761 | 1.000 | 306 | $5,000 | 8 sigs, 203 writeups |
| 3 | [CWE-79 Improper Neutralization of Input During Web ](https://cwe.mitre.org/data/definitions/79.html) | 0.743 | 0.774 | 154 | $1,250 | 8 sigs, 1184 writeups |
| 4 | [CWE-502 Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html) | 0.691 | 1.000 | 83 | $4,500 | 8 sigs, 114 writeups |
| 5 | [CWE-94 Improper Control of Generation of Code ('Cod](https://cwe.mitre.org/data/definitions/94.html) | 0.556 | 0.977 | 49 | $5,600 | 3 sigs, 880 writeups |

### What to try: CWE-200 Exposure of Sensitive Information to an Unauthorized Actor
- **Explore** — [Request Footprinting] The attacker examines the website information and source code of the website and uses automated tools to get as much information as possible about the system and organization.
    - Open Source Footprinting: Examine the website about the organization and skim through the webpage's HTML source to look for comments.
    - Network Enumeration: Perform various queries (Registrar Query, Organizational Query, Domain Query, Network Query, POC Query) on the many whois databases found on the internet to identify domain names and associated networks.
    - DNS Interrogation: Once basic information is gathered the attack could begin to query DNS.
    - Other Techniques: Use ping sweep, TCP scan, UDP scan, OS Identification various techniques to gain more information about the system and network.
    - _source: [CAPEC-169: Footprinting](https://capec.mitre.org/data/definitions/169.html)_
- **Explore** — [Find Session IDs] The attacker interacts with the target host and finds that session IDs are used to authenticate users.
    - An attacker makes many anonymous connections and records the session IDs assigned.
    - An attacker makes authorized connections and records the session tokens or credentials issued.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_
- **Explore** — [Characterize IDs] The attacker studies the characteristics of the session ID (size, format, etc.). As a results the attacker finds that legitimate session IDs are predictable.
    - Cryptanalysis. The attacker uses cryptanalysis to determine if the session IDs contain any cryptographic protections.
    - Pattern tests. The attacker looks for patterns (odd/even, repetition, multiples, or other arithmetic relationships) between IDs
    - Comparison against time. The attacker plots or compares the issued IDs to the time they were issued to check for correlation.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_
- **Experiment** — [Match issued IDs] The attacker brute forces different values of session ID and manages to predict a valid session ID.
    - The attacker models the session ID algorithm enough to produce a compatible session IDs, or just one match.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_

**Detection templates to run:**
- [Jira Server and Data Center - Information Disclosure](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-14181.yaml) (medium)
- [Apache Tomcat - Default Login Discovery](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/default-logins/apache/tomcat-examples-login.yaml) (info)
- [Jira Service Desk Login Panel - Detect](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/exposed-panels/servicedesk-login-panel.yaml) (info)

**Prior disclosures:**
- [Exploiting CSP in Webkit to Break Authentication & Authorization](https://threatnix.io/blog/exploiting-csp-in-webkit-to-break-authentication-authorization/) — $100,000
- [Instagram App Access Token](https://philippeharewood.com/instagram-app-access-token/) — $38,300
- [How I made more than $30K with Jolokia CVEs](https://wss.sh/en/blog/how-i-made-more-than-30k-with-jolokia-cves/) — $33,500

### What to try: CWE-22 Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser, an automated tool or by inspecting the application, an adversary records all entry points to the application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all user input entry points visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - Manually inspect the application to find entry points.
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Explore** — The attacker accesses the server using a specific URL.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_
- **Experiment** — [Probe entry points to locate vulnerabilities] The adversary uses the entry points gathered in the "Explore" phase as a target list and looks for areas where user input is used to access resources on the target host. The adversary attempts different encodings of slash characters to bypass input filters.
    - Try both backslash and forward slash characters
    - Try different encodings for slash characters such as %5C
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Experiment** — The attacker tries to encode some special characters in the URL. The attacker find out that some characters are not filtered properly.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_

**Detection templates to run:**
- [Atlassian Jira Limited - Local File Inclusion](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2021/CVE-2021-26086.yaml) (medium)
- [SAP NetWeaver Application Server Java 7.5 - Local File Inclusion](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2017/CVE-2017-12637.yaml) (high)
- [Apache Tomcat JK Connect <=1.2.44 - Manager Access](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2018/CVE-2018-11759.yaml) (high)

**Prior disclosures:**
- [GitHub Pages - Multiple RCEs via insecure Kramdown configuration - $25,000 Bount](https://devcraft.io/2020/10/20/github-pages-multiple-rces-via-kramdown-config.html) — $25,000
- [CS:GO: From Zero to 0-day](https://neodyme.io/blog/csgo_from_zero_to_0day/) — $22,500
- [Two weeks of securing Samsung devices: Part 2](https://blog.oversecured.com/Two-weeks-of-securing-Samsung-devices-Part-2/) — $18,040

### What to try: CWE-79 Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser or an automated tool, an attacker follows all public links and actions on a web site. They record all the links, the forms, the resources accessed and all other potential entry-points for the web application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all links visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_
- **Explore** — [Send request to target webpage and analyze HTML] Using a browser or an automated tool, an adversary sends requests to a webpage and records the received HTML response. Adversaries then analyze the HTML to identify any known underlying JavaScript architectures. This can aid in mappiong publicly known vulnerabilities to the webpage and can also helpo the adversary guess application architecture and the inner workings of a system.
    - Record all "src" values inside script tags. These JavaScript files are compared to lists of files for known architectures. If there is a large match between the "src" values and architecture files, then it can be assumed that particular architecture is being used.
    - _source: [CAPEC-85: AJAX Footprinting](https://capec.mitre.org/data/definitions/85.html)_
- **Experiment** — [Probe identified potential entry points for XSS vulnerability] The attacker uses the entry points gathered in the "Explore" phase as a target list and injects various common script payloads to determine if an entry point actually represents a vulnerability and to characterize the extent to which the vulnerability can be exploited.
    - Use a list of XSS probe strings to inject script in parameters of known URLs. If possible, the probe strings contain a unique identifier.
    - Use a proxy tool to record results of manual input of XSS probes in known URLs.
    - Use a list of XSS probe strings to inject script into UI entry fields. If possible, the probe strings contain a unique identifier.
    - Use a list of XSS probe strings to inject script into resources accessed by the application. If possible, the probe strings contain a unique identifier.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_

**Detection templates to run:**
- [Apache Tomcat 4.x-7.x - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2007/CVE-2007-2449.yaml) (medium)
- [Apache Tomcat - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2019/CVE-2019-0221.yaml) (medium)
- [Atlassian Jira WallboardServlet <7.13.1 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2018/CVE-2018-20824.yaml) (medium)

**Prior disclosures:**
- [We Hacked Apple for 3 Months: Here’s What We Found](https://samcurry.net/hacking-apple/) — $288,500
- [How I Hacked Microsoft Teams and got $150,000 in Pwn2Own](https://speakerdeck.com/masatokinugawa/how-i-hacked-microsoft-teams-and-got-150000-dollars-in-pwn2own) — $150,000
- [How we made $120k bug bounty in a year with good automation](https://www.vidocsecurity.com/blog/2022-summary-how-we-made-120k-bug-bounty-in-a-year/) — $120,000

## ci.acme.example
**Fingerprinted:** Jenkins, Jetty

| # | Weakness | Score | EPSS | Exploits | Median bounty | Signals |
|---|----------|-------|------|----------|---------------|---------|
| 1 | [CWE-200 Exposure of Sensitive Information to an Unau](https://cwe.mitre.org/data/definitions/200.html) | 0.886 | 0.993 | 23 | $1,500 | 8 sigs, 672 writeups |
| 2 | [CWE-79 Improper Neutralization of Input During Web ](https://cwe.mitre.org/data/definitions/79.html) | 0.618 | 0.894 | 154 | $1,250 | 4 sigs, 1184 writeups |
| 3 | [CWE-502 Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html) | 0.477 | 0.997 | 83 | $4,500 | 2 sigs, 114 writeups |
| 4 | [CWE-78 Improper Neutralization of Special Elements ](https://cwe.mitre.org/data/definitions/78.html) | 0.427 | — | 80 | $4,000 | 1 sigs, 104 writeups |
| 5 | [CWE-94 Improper Control of Generation of Code ('Cod](https://cwe.mitre.org/data/definitions/94.html) | 0.391 | — | 49 | $5,600 | 1 sigs, 880 writeups |

### What to try: CWE-200 Exposure of Sensitive Information to an Unauthorized Actor
- **Explore** — [Request Footprinting] The attacker examines the website information and source code of the website and uses automated tools to get as much information as possible about the system and organization.
    - Open Source Footprinting: Examine the website about the organization and skim through the webpage's HTML source to look for comments.
    - Network Enumeration: Perform various queries (Registrar Query, Organizational Query, Domain Query, Network Query, POC Query) on the many whois databases found on the internet to identify domain names and associated networks.
    - DNS Interrogation: Once basic information is gathered the attack could begin to query DNS.
    - Other Techniques: Use ping sweep, TCP scan, UDP scan, OS Identification various techniques to gain more information about the system and network.
    - _source: [CAPEC-169: Footprinting](https://capec.mitre.org/data/definitions/169.html)_
- **Explore** — [Find Session IDs] The attacker interacts with the target host and finds that session IDs are used to authenticate users.
    - An attacker makes many anonymous connections and records the session IDs assigned.
    - An attacker makes authorized connections and records the session tokens or credentials issued.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_
- **Explore** — [Characterize IDs] The attacker studies the characteristics of the session ID (size, format, etc.). As a results the attacker finds that legitimate session IDs are predictable.
    - Cryptanalysis. The attacker uses cryptanalysis to determine if the session IDs contain any cryptographic protections.
    - Pattern tests. The attacker looks for patterns (odd/even, repetition, multiples, or other arithmetic relationships) between IDs
    - Comparison against time. The attacker plots or compares the issued IDs to the time they were issued to check for correlation.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_
- **Experiment** — [Match issued IDs] The attacker brute forces different values of session ID and manages to predict a valid session ID.
    - The attacker models the session ID algorithm enough to produce a compatible session IDs, or just one match.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_

**Detection templates to run:**
- [Eclipse Jetty - Information Disclosure](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2021/CVE-2021-34429.yaml) (medium)
- [Jenkins GitHub Plugin <=1.29.1 - Server-Side Request Forgery](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2018/CVE-2018-1000600.yaml) (high)
- [Eclipse Jetty - Information Disclosure](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2021/CVE-2021-28164.yaml) (medium)

**Prior disclosures:**
- [Exploiting CSP in Webkit to Break Authentication & Authorization](https://threatnix.io/blog/exploiting-csp-in-webkit-to-break-authentication-authorization/) — $100,000
- [Instagram App Access Token](https://philippeharewood.com/instagram-app-access-token/) — $38,300
- [How I made more than $30K with Jolokia CVEs](https://wss.sh/en/blog/how-i-made-more-than-30k-with-jolokia-cves/) — $33,500

### What to try: CWE-79 Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser or an automated tool, an attacker follows all public links and actions on a web site. They record all the links, the forms, the resources accessed and all other potential entry-points for the web application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all links visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_
- **Explore** — [Send request to target webpage and analyze HTML] Using a browser or an automated tool, an adversary sends requests to a webpage and records the received HTML response. Adversaries then analyze the HTML to identify any known underlying JavaScript architectures. This can aid in mappiong publicly known vulnerabilities to the webpage and can also helpo the adversary guess application architecture and the inner workings of a system.
    - Record all "src" values inside script tags. These JavaScript files are compared to lists of files for known architectures. If there is a large match between the "src" values and architecture files, then it can be assumed that particular architecture is being used.
    - _source: [CAPEC-85: AJAX Footprinting](https://capec.mitre.org/data/definitions/85.html)_
- **Experiment** — [Probe identified potential entry points for XSS vulnerability] The attacker uses the entry points gathered in the "Explore" phase as a target list and injects various common script payloads to determine if an entry point actually represents a vulnerability and to characterize the extent to which the vulnerability can be exploited.
    - Use a list of XSS probe strings to inject script in parameters of known URLs. If possible, the probe strings contain a unique identifier.
    - Use a proxy tool to record results of manual input of XSS probes in known URLs.
    - Use a list of XSS probe strings to inject script into UI entry fields. If possible, the probe strings contain a unique identifier.
    - Use a list of XSS probe strings to inject script into resources accessed by the application. If possible, the probe strings contain a unique identifier.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_

**Detection templates to run:**
- [Jenkins Gitlab Hook <=1.4.2 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-2096.yaml) (medium)
- [Jenkin Audit Trail <=3.2 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-2140.yaml) (medium)
- [Jenkins <=2.196 - Cookie Exposure](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2019/CVE-2019-10405.yaml) (medium)

**Prior disclosures:**
- [We Hacked Apple for 3 Months: Here’s What We Found](https://samcurry.net/hacking-apple/) — $288,500
- [How I Hacked Microsoft Teams and got $150,000 in Pwn2Own](https://speakerdeck.com/masatokinugawa/how-i-hacked-microsoft-teams-and-got-150000-dollars-in-pwn2own) — $150,000
- [How we made $120k bug bounty in a year with good automation](https://www.vidocsecurity.com/blog/2022-summary-how-we-made-120k-bug-bounty-in-a-year/) — $120,000

### What to try: CWE-502 Deserialization of Untrusted Data
- _No CAPEC execution flow recorded for this class._

**Detection templates to run:**
- [Jenkins CLI - Java Deserialization](https://github.com/projectdiscovery/nuclei-templates/blob/main/code/cves/2017/CVE-2017-1000353.yaml) (critical)
- [Jenkins - Remote Command Injection](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2018/CVE-2018-1000861.yaml) (critical)

**Prior disclosures:**
- [Hello Lucee! Let us hack Apple again?](https://blog.projectdiscovery.io/hello-lucee-let-us-hack-apple-again/) — $20,000
- [How I Chained 4 vulnerabilities on GitHub Enterprise, From SSRF Execution Chain ](https://blog.orange.tw/2017/07/how-i-chained-4-vulnerabilities-on.html) — $12,500
- [$4,998 Bounty Awarded and 100,000 WordPress Sites Protected Against Unauthentica](https://www.wordfence.com/blog/2024/08/4998-bounty-awarded-and-100000-wordpress-sites-protected-against-unauthenticated-remote-code-execution-vulnerability-patched-in-givewp-wordpress-plugin/) — $4,998

## api.acme.example
**Fingerprinted:** Express, GraphQL, Node.js

| # | Weakness | Score | EPSS | Exploits | Median bounty | Signals |
|---|----------|-------|------|----------|---------------|---------|
| 1 | [CWE-22 Improper Limitation of a Pathname to a Restr](https://cwe.mitre.org/data/definitions/22.html) | 0.847 | 0.721 | 306 | $5,000 | 8 sigs, 203 writeups |
| 2 | [CWE-200 Exposure of Sensitive Information to an Unau](https://cwe.mitre.org/data/definitions/200.html) | 0.790 | 0.349 | 23 | $1,500 | 4 sigs, 672 writeups |
| 3 | [CWE-79 Improper Neutralization of Input During Web ](https://cwe.mitre.org/data/definitions/79.html) | 0.649 | 0.151 | 154 | $1,250 | 5 sigs, 1184 writeups |
| 4 | [CWE-78 Improper Neutralization of Special Elements ](https://cwe.mitre.org/data/definitions/78.html) | 0.543 | 0.907 | 80 | $4,000 | 2 sigs, 104 writeups |
| 5 | [CWE-89 Improper Neutralization of Special Elements ](https://cwe.mitre.org/data/definitions/89.html) | 0.505 | 0.693 | 57 | $3,000 | 1 sigs, 230 writeups |

### What to try: CWE-22 Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser, an automated tool or by inspecting the application, an adversary records all entry points to the application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all user input entry points visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - Manually inspect the application to find entry points.
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Explore** — The attacker accesses the server using a specific URL.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_
- **Experiment** — [Probe entry points to locate vulnerabilities] The adversary uses the entry points gathered in the "Explore" phase as a target list and looks for areas where user input is used to access resources on the target host. The adversary attempts different encodings of slash characters to bypass input filters.
    - Try both backslash and forward slash characters
    - Try different encodings for slash characters such as %5C
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Experiment** — The attacker tries to encode some special characters in the URL. The attacker find out that some characters are not filtered properly.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_

**Detection templates to run:**
- [Totaljs <3.2.3 - Local File Inclusion](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2019/CVE-2019-8903.yaml) (high)
- [Node.js <8.6.0 - Directory Traversal](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2017/CVE-2017-14849.yaml) (high)
- [Ghost CMS < 5.42.1 - Path Traversal](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2023/CVE-2023-32235.yaml) (high)

**Prior disclosures:**
- [GitHub Pages - Multiple RCEs via insecure Kramdown configuration - $25,000 Bount](https://devcraft.io/2020/10/20/github-pages-multiple-rces-via-kramdown-config.html) — $25,000
- [CS:GO: From Zero to 0-day](https://neodyme.io/blog/csgo_from_zero_to_0day/) — $22,500
- [Two weeks of securing Samsung devices: Part 2](https://blog.oversecured.com/Two-weeks-of-securing-Samsung-devices-Part-2/) — $18,040

### What to try: CWE-200 Exposure of Sensitive Information to an Unauthorized Actor
- **Explore** — [Request Footprinting] The attacker examines the website information and source code of the website and uses automated tools to get as much information as possible about the system and organization.
    - Open Source Footprinting: Examine the website about the organization and skim through the webpage's HTML source to look for comments.
    - Network Enumeration: Perform various queries (Registrar Query, Organizational Query, Domain Query, Network Query, POC Query) on the many whois databases found on the internet to identify domain names and associated networks.
    - DNS Interrogation: Once basic information is gathered the attack could begin to query DNS.
    - Other Techniques: Use ping sweep, TCP scan, UDP scan, OS Identification various techniques to gain more information about the system and network.
    - _source: [CAPEC-169: Footprinting](https://capec.mitre.org/data/definitions/169.html)_
- **Explore** — [Find Session IDs] The attacker interacts with the target host and finds that session IDs are used to authenticate users.
    - An attacker makes many anonymous connections and records the session IDs assigned.
    - An attacker makes authorized connections and records the session tokens or credentials issued.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_
- **Explore** — [Characterize IDs] The attacker studies the characteristics of the session ID (size, format, etc.). As a results the attacker finds that legitimate session IDs are predictable.
    - Cryptanalysis. The attacker uses cryptanalysis to determine if the session IDs contain any cryptographic protections.
    - Pattern tests. The attacker looks for patterns (odd/even, repetition, multiples, or other arithmetic relationships) between IDs
    - Comparison against time. The attacker plots or compares the issued IDs to the time they were issued to check for correlation.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_
- **Experiment** — [Match issued IDs] The attacker brute forces different values of session ID and manages to predict a valid session ID.
    - The attacker models the session ID algorithm enough to produce a compatible session IDs, or just one match.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_

**Detection templates to run:**
- [Gitlab CE/EE 13.4 - 13.6.2 - Information Disclosure](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-26413.yaml) (medium)
- [Express-handlebars - Local File Inclusion](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2021/CVE-2021-32820.yaml) (high)
- [SuiteCRM Unauthenticated Graphql Introspection](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2023/CVE-2023-47643.yaml) (medium)

**Prior disclosures:**
- [Exploiting CSP in Webkit to Break Authentication & Authorization](https://threatnix.io/blog/exploiting-csp-in-webkit-to-break-authentication-authorization/) — $100,000
- [Instagram App Access Token](https://philippeharewood.com/instagram-app-access-token/) — $38,300
- [How I made more than $30K with Jolokia CVEs](https://wss.sh/en/blog/how-i-made-more-than-30k-with-jolokia-cves/) — $33,500

### What to try: CWE-79 Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser or an automated tool, an attacker follows all public links and actions on a web site. They record all the links, the forms, the resources accessed and all other potential entry-points for the web application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all links visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_
- **Explore** — [Send request to target webpage and analyze HTML] Using a browser or an automated tool, an adversary sends requests to a webpage and records the received HTML response. Adversaries then analyze the HTML to identify any known underlying JavaScript architectures. This can aid in mappiong publicly known vulnerabilities to the webpage and can also helpo the adversary guess application architecture and the inner workings of a system.
    - Record all "src" values inside script tags. These JavaScript files are compared to lists of files for known architectures. If there is a large match between the "src" values and architecture files, then it can be assumed that particular architecture is being used.
    - _source: [CAPEC-85: AJAX Footprinting](https://capec.mitre.org/data/definitions/85.html)_
- **Experiment** — [Probe identified potential entry points for XSS vulnerability] The attacker uses the entry points gathered in the "Explore" phase as a target list and injects various common script payloads to determine if an entry point actually represents a vulnerability and to characterize the extent to which the vulnerability can be exploited.
    - Use a list of XSS probe strings to inject script in parameters of known URLs. If possible, the probe strings contain a unique identifier.
    - Use a proxy tool to record results of manual input of XSS probes in known URLs.
    - Use a list of XSS probe strings to inject script into UI entry fields. If possible, the probe strings contain a unique identifier.
    - Use a list of XSS probe strings to inject script into resources accessed by the application. If possible, the probe strings contain a unique identifier.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_

**Detection templates to run:**
- [karma-runner DOM-based Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2022/CVE-2022-0437.yaml) (medium)
- [npm ansi_up v4 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2021/CVE-2021-3377.yaml) (medium)
- [Ghost CMS <=4.32 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2021/CVE-2021-29484.yaml) (medium)

**Prior disclosures:**
- [We Hacked Apple for 3 Months: Here’s What We Found](https://samcurry.net/hacking-apple/) — $288,500
- [How I Hacked Microsoft Teams and got $150,000 in Pwn2Own](https://speakerdeck.com/masatokinugawa/how-i-hacked-microsoft-teams-and-got-150000-dollars-in-pwn2own) — $150,000
- [How we made $120k bug bounty in a year with good automation](https://www.vidocsecurity.com/blog/2022-summary-how-we-made-120k-bug-bounty-in-a-year/) — $120,000

## git.acme.example
**Fingerprinted:** GitLab, nginx
**CVEs found by recon:** CVE-2021-22205

| # | Weakness | Score | EPSS | Exploits | Median bounty | Signals |
|---|----------|-------|------|----------|---------------|---------|
| 1 | [CWE-200 Exposure of Sensitive Information to an Unau](https://cwe.mitre.org/data/definitions/200.html) | 0.790 | 0.349 | 23 | $1,500 | 7 sigs, 672 writeups |
| 2 | [CWE-22 Improper Limitation of a Pathname to a Restr](https://cwe.mitre.org/data/definitions/22.html) | 0.553 | 0.716 | 306 | $5,000 | 3 sigs, 203 writeups |
| 3 | [CWE-79 Improper Neutralization of Input During Web ](https://cwe.mitre.org/data/definitions/79.html) | 0.522 | 0.894 | 154 | $1,250 | 1 sigs, 1184 writeups |
| 4 | [CWE-78 Improper Neutralization of Special Elements ](https://cwe.mitre.org/data/definitions/78.html) | 0.495 | 0.769 | 80 | $4,000 | 2 sigs, 104 writeups |
| 5 | [CWE-94 Improper Control of Generation of Code ('Cod](https://cwe.mitre.org/data/definitions/94.html) | 0.492 | 0.997 | 49 | $5,600 | 1 sigs, 880 writeups |

### What to try: CWE-200 Exposure of Sensitive Information to an Unauthorized Actor
- **Explore** — [Request Footprinting] The attacker examines the website information and source code of the website and uses automated tools to get as much information as possible about the system and organization.
    - Open Source Footprinting: Examine the website about the organization and skim through the webpage's HTML source to look for comments.
    - Network Enumeration: Perform various queries (Registrar Query, Organizational Query, Domain Query, Network Query, POC Query) on the many whois databases found on the internet to identify domain names and associated networks.
    - DNS Interrogation: Once basic information is gathered the attack could begin to query DNS.
    - Other Techniques: Use ping sweep, TCP scan, UDP scan, OS Identification various techniques to gain more information about the system and network.
    - _source: [CAPEC-169: Footprinting](https://capec.mitre.org/data/definitions/169.html)_
- **Explore** — [Find Session IDs] The attacker interacts with the target host and finds that session IDs are used to authenticate users.
    - An attacker makes many anonymous connections and records the session IDs assigned.
    - An attacker makes authorized connections and records the session tokens or credentials issued.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_
- **Explore** — [Characterize IDs] The attacker studies the characteristics of the session ID (size, format, etc.). As a results the attacker finds that legitimate session IDs are predictable.
    - Cryptanalysis. The attacker uses cryptanalysis to determine if the session IDs contain any cryptographic protections.
    - Pattern tests. The attacker looks for patterns (odd/even, repetition, multiples, or other arithmetic relationships) between IDs
    - Comparison against time. The attacker plots or compares the issued IDs to the time they were issued to check for correlation.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_
- **Experiment** — [Match issued IDs] The attacker brute forces different values of session ID and manages to predict a valid session ID.
    - The attacker models the session ID algorithm enough to produce a compatible session IDs, or just one match.
    - _source: [CAPEC-59: Session Credential Falsification through Prediction](https://capec.mitre.org/data/definitions/59.html)_

**Detection templates to run:**
- [Gitlab CE/EE 13.4 - 13.6.2 - Information Disclosure](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-26413.yaml) (medium)
- [Nginx UI Panel - Detect](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/exposed-panels/nginx-ui-dashboard.yaml) (info)
- [Nginx Admin Manager Login Panel - Detect](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/exposed-panels/nginx-admin-panel.yaml) (info)

**Prior disclosures:**
- [Exploiting CSP in Webkit to Break Authentication & Authorization](https://threatnix.io/blog/exploiting-csp-in-webkit-to-break-authentication-authorization/) — $100,000
- [Instagram App Access Token](https://philippeharewood.com/instagram-app-access-token/) — $38,300
- [How I made more than $30K with Jolokia CVEs](https://wss.sh/en/blog/how-i-made-more-than-30k-with-jolokia-cves/) — $33,500

### What to try: CWE-22 Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser, an automated tool or by inspecting the application, an adversary records all entry points to the application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all user input entry points visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - Manually inspect the application to find entry points.
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Explore** — The attacker accesses the server using a specific URL.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_
- **Experiment** — [Probe entry points to locate vulnerabilities] The adversary uses the entry points gathered in the "Explore" phase as a target list and looks for areas where user input is used to access resources on the target host. The adversary attempts different encodings of slash characters to bypass input filters.
    - Try both backslash and forward slash characters
    - Try different encodings for slash characters such as %5C
    - _source: [CAPEC-79: Using Slashes in Alternate Encoding](https://capec.mitre.org/data/definitions/79.html)_
- **Experiment** — The attacker tries to encode some special characters in the URL. The attacker find out that some characters are not filtered properly.
    - _source: [CAPEC-64: Using Slashes and URL Encoding Combined to Bypass Validation Logic](https://capec.mitre.org/data/definitions/64.html)_

**Detection templates to run:**
- [GitLab 16.0.0 - Path Traversal](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2023/CVE-2023-2825.yaml) (high)
- [Nginx Server - Local File Inclusion](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/vulnerabilities/other/nginx-merge-slashes-path-traversal.yaml) (high)
- [Nginx Status Page - 403 Bypass](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/misconfiguration/nginx/nginx-status-403-bypass.yaml) (low)

**Prior disclosures:**
- [GitHub Pages - Multiple RCEs via insecure Kramdown configuration - $25,000 Bount](https://devcraft.io/2020/10/20/github-pages-multiple-rces-via-kramdown-config.html) — $25,000
- [CS:GO: From Zero to 0-day](https://neodyme.io/blog/csgo_from_zero_to_0day/) — $22,500
- [Two weeks of securing Samsung devices: Part 2](https://blog.oversecured.com/Two-weeks-of-securing-Samsung-devices-Part-2/) — $18,040

### What to try: CWE-79 Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
- **Explore** — [Survey the application for user-controllable inputs] Using a browser or an automated tool, an attacker follows all public links and actions on a web site. They record all the links, the forms, the resources accessed and all other potential entry-points for the web application.
    - Use a spidering tool to follow and record all links and analyze the web pages to find entry points. Make special note of any links that include parameters in the URL.
    - Use a proxy tool to record all links visited during a manual traversal of the web application.
    - Use a browser to manually explore the website and analyze how it is constructed. Many browsers' plugins are available to facilitate the analysis or automate the discovery.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_
- **Explore** — [Send request to target webpage and analyze HTML] Using a browser or an automated tool, an adversary sends requests to a webpage and records the received HTML response. Adversaries then analyze the HTML to identify any known underlying JavaScript architectures. This can aid in mappiong publicly known vulnerabilities to the webpage and can also helpo the adversary guess application architecture and the inner workings of a system.
    - Record all "src" values inside script tags. These JavaScript files are compared to lists of files for known architectures. If there is a large match between the "src" values and architecture files, then it can be assumed that particular architecture is being used.
    - _source: [CAPEC-85: AJAX Footprinting](https://capec.mitre.org/data/definitions/85.html)_
- **Experiment** — [Probe identified potential entry points for XSS vulnerability] The attacker uses the entry points gathered in the "Explore" phase as a target list and injects various common script payloads to determine if an entry point actually represents a vulnerability and to characterize the extent to which the vulnerability can be exploited.
    - Use a list of XSS probe strings to inject script in parameters of known URLs. If possible, the probe strings contain a unique identifier.
    - Use a proxy tool to record results of manual input of XSS probes in known URLs.
    - Use a list of XSS probe strings to inject script into UI entry fields. If possible, the probe strings contain a unique identifier.
    - Use a list of XSS probe strings to inject script into resources accessed by the application. If possible, the probe strings contain a unique identifier.
    - _source: [CAPEC-63: Cross-Site Scripting (XSS)](https://capec.mitre.org/data/definitions/63.html)_

**Detection templates to run:**
- [Jenkins Gitlab Hook <=1.4.2 - Cross-Site Scripting](https://github.com/projectdiscovery/nuclei-templates/blob/main/http/cves/2020/CVE-2020-2096.yaml) (medium)

**Prior disclosures:**
- [We Hacked Apple for 3 Months: Here’s What We Found](https://samcurry.net/hacking-apple/) — $288,500
- [How I Hacked Microsoft Teams and got $150,000 in Pwn2Own](https://speakerdeck.com/masatokinugawa/how-i-hacked-microsoft-teams-and-got-150000-dollars-in-pwn2own) — $150,000
- [How we made $120k bug bounty in a year with good automation](https://www.vidocsecurity.com/blog/2022-summary-how-we-made-120k-bug-bounty-in-a-year/) — $120,000
