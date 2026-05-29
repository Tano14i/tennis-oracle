"""
deploy_to_github.py
Pubblica tennis_oracle_mobile.html su GitHub Pages automaticamente.
Esegui una volta sola dal PC:
    python deploy_to_github.py
"""

import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Tano14i/tennis-oracle.git"
SOURCE_FILE = "tennis_oracle_mobile.html"

# Verifica che il file esista
if not os.path.exists(SOURCE_FILE):
    print(f"ERRORE: {SOURCE_FILE} non trovato.")
    print("Metti deploy_to_github.py nella stessa cartella di tennis_oracle_mobile.html")
    sys.exit(1)

def run(cmd, cwd=None):
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0 and result.stderr:
        print(f"    ERRORE: {result.stderr.strip()}")
    return result.returncode

print("=== Deploy Tennis Oracle su GitHub Pages ===")
print()

# Verifica git installato
if run("git --version") != 0:
    print("Git non trovato. Scaricalo da https://git-scm.com/download/win")
    sys.exit(1)

# Clona il repo in una cartella temporanea
deploy_dir = "_deploy_temp"
if os.path.exists(deploy_dir):
    shutil.rmtree(deploy_dir)

print("1. Clono il repository...")
if run(f'git clone {REPO_URL} {deploy_dir}') != 0:
    print("Errore nel clone. Verifica la connessione internet.")
    sys.exit(1)

# Copia il file come index.html
print("2. Copio il file come index.html...")
shutil.copy2(SOURCE_FILE, os.path.join(deploy_dir, "index.html"))

# Configura git e fa il push
print("3. Pubblico su GitHub...")
cwd = deploy_dir

run('git config user.email "deploy@tennsisoracle.com"', cwd=cwd)
run('git config user.name "Tennis Oracle Deploy"', cwd=cwd)
run('git add index.html', cwd=cwd)
run('git commit -m "Update Tennis Oracle app"', cwd=cwd)

push_result = run('git push', cwd=cwd)

# Pulizia
shutil.rmtree(deploy_dir)

if push_result == 0:
    print()
    print("=" * 50)
    print("SUCCESSO! App pubblicata.")
    print()
    print("Ora attiva GitHub Pages:")
    print("  1. Vai su https://github.com/Tano14i/tennis-oracle/settings/pages")
    print("  2. Source -> Deploy from branch -> main -> Save")
    print()
    print("Dopo 1-2 minuti la tua app e disponibile su:")
    print("  https://tano14i.github.io/tennis-oracle")
    print()
    print("Salva il link come segnalibro sul telefono.")
else:
    print()
    print("Errore nel push. Probabilmente serve autenticazione.")
    print("Soluzione:")
    print("  1. Vai su GitHub -> Settings -> Developer settings -> Personal access tokens")
    print("  2. Genera un token con permesso 'repo'")
    print("  3. Quando git chiede la password, usa il token")
