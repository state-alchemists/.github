# Moving WSL from Drive C to Drive D

Motivation:
- Since 2020, I no longer partition my harddisk into C and D.
    - I believe OneDrive is sufficient
    - I mostly works with WSL, and I want to use as much disk as possible
- For professional work, I need to use a laptop with C and D partition.
    - The D drive is underutilized
    - The C drive is full

Drawback:
- Installing WSL on drive D seems to be slower than installing it on Drive C


# How to

First, I need to export the existing `ubuntu` distro. I run the following script in powershell.

```bash
D:
mkdir WSL
mkdir WSL\Ubuntu-22.04
wsl.exe --export ubuntu D:\WSL\Ubuntu-22.04\Ubuntu-22.04.tar
```

Then, I need to create a new distro named `ubuntud`.

```bash
mkdir WSL\Ubuntu2204
wsl.exe --import ubuntud D:\WSL\Ubuntu2204\ D:\WSL\Ubuntu22.04\Ubuntu22.04.tar
```

Before I go any further, I need to confirm that `ubuntud` is accessible

```bash
wsl -d ubuntud
```

This works, but the default user for `ubuntud` is now `root`. I need to change it back.

```bash
# You can also use nano, if you are not comfortable with vim
vim /etc/wsl.conf
```

I change the default user name to `gofrendi`

```
[user]
default=gofrendi
```

After saving the file, I quit from `ubuntud` terminal and terminate the distro.

```bash
exit
wsl --terminate ubuntud
```

To make sure again, I enter `ubuntud` terminal, confirming that the default user is now `gofrendi`.

```bash
wsl -d ubuntud
```

Once everything is okay, I set the default distro to `ubuntud` and remove `ubuntu`:

```bash
wsl --set-default ubuntud
wsl --unregister ubuntu
```