#  **NMON ANALYSER FOR LINUX MACHINE WITH PYTHON**


This project automates the process of collecting, analyzing, and reporting system performance data from remote servers using NMON files. The tool performs the following tasks:

- Connects to remote hosts via SSH to fetch NMON files for specified dates.
- Analyzes the data to generate insights and plots for key system metrics.
- Creates a detailed report in `.docx` format for each host and date combination.
- Sends email notifications with the generated reports as attachments.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Output](#output)
7. [Email Integration](#email-integration)
8. [License](#license)

---

## Features

- Fetch NMON files for specific hosts and dates via SSH.
- Analyze system performance metrics, including:
  - CPU Usage
  - Memory Usage
  - Network I/O
  - Disk I/O and File Systems
- Generate detailed `.docx` reports with visual graphs.
- Automatically email generated reports for all hosts.

---

## Prerequisites

Ensure the following tools and libraries are installed:

1. **Python 3.6+**
2. **Required Python Libraries**: 
   - `paramiko`
   - `matplotlib`
   - `pandas`
   - `docx` (via `python-docx`)
   - `argparse`
3. **Sendmail Service** for SMTP configuration (no authentication required).

Ensure the Sendmail service is properly configured on your system.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mtptisid/NMON_Analyser_with_Python.git
   cd NMON_Analyser_with_Python
    ```


## Snapshots
![image](https://github.com/user-attachments/assets/1fcf31d6-b6a7-4e41-aa7b-b6a71c9c9677)

