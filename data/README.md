# Data Assets

This folder tracks the datasets used by the project without committing large external archives.

Run the downloader from the repository root:

```powershell
.\data\download.ps1
```

By default, the script downloads reasonably sized public assets and skips very large archives.
Use `-IncludeLarge` to include the full Project CodeNet archive.

```powershell
.\data\download.ps1 -IncludeLarge
```

Use `-GenerateSynthetic` to create the two local synthetic datasets:

```powershell
.\data\download.ps1 -GenerateSynthetic
```

Downloaded files are written to `data/raw/`. Generated datasets are written to `data/generated/`.

## Dataset Coverage

| Dataset / Data Source | Type | Handling |
| --- | --- | --- |
| X-COBOL / public GitHub COBOL repositories | General COBOL code corpus | Downloaded from Zenodo record 14269462 |
| IBM Project CodeNet - COBOL subset | General programming dataset | Full CodeNet archive is listed as a large optional download; filter after extraction |
| IBM CICS Banking Sample Application / CBSA | Banking COBOL application | Downloaded from GitHub |
| AWS Mainframe Modernization CardDemo | Credit-card COBOL application | Downloaded from GitHub |
| RBI KYC Master Direction | Financial regulation corpus | Downloaded from RBI PDF |
| Basel Framework / Basel III | Banking regulation corpus | Downloaded from BIS PDF |
| Generated COBOL Logic-Block Dataset | Processed dataset | Generated locally by `scripts/generate_synthetic_datasets.py` |
| Business Intent Card Dataset | Instruction-tuning / evaluation dataset | Generated locally by `scripts/generate_synthetic_datasets.py` |

## Notes

- Check the upstream license or terms for each source before redistributing derived artifacts.
- Project CodeNet is large. The public DAX archive is about 7.8 GB compressed, so it is skipped unless `-IncludeLarge` is provided.
- The CodeNet public archive is not a pre-cut COBOL-only archive. After downloading and extracting it, filter files by the `COBOL` language directory or CodeNet metadata if present in the version you download.
