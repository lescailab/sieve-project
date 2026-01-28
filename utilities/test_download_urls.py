#!/usr/bin/env python3
"""
Test script to verify download URLs for database utilities.

This script checks that all download URLs are accessible before
attempting full downloads.

Usage:
    python utilities/test_download_urls.py
"""

import sys
import urllib.request
import urllib.error


def test_url(name: str, url: str) -> bool:
    """Test if a URL is accessible (returns 200 or 302)."""
    print(f"\nTesting {name}:")
    print(f"  URL: {url}")

    try:
        req = urllib.request.Request(url, method='HEAD')
        response = urllib.request.urlopen(req, timeout=10)
        status = response.getcode()

        if status in [200, 302]:
            print(f"  ✓ SUCCESS (HTTP {status})")
            return True
        else:
            print(f"  ✗ UNEXPECTED STATUS: HTTP {status}")
            return False

    except urllib.error.HTTPError as e:
        print(f"  ✗ FAILED: HTTP {e.code} {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"  ✗ FAILED: {e.reason}")
        return False
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def main():
    """Test all database download URLs."""
    print("="*70)
    print("Database Download URL Verification")
    print("="*70)

    urls_to_test = {
        'GWAS Catalog API': 'https://www.ebi.ac.uk/gwas/api/search/downloads/associations/v1.0?split=false',
        'ClinVar GRCh37 VCF': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/clinvar.vcf.gz',
        'ClinVar GRCh38 VCF': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz',
    }

    results = {}
    for name, url in urls_to_test.items():
        results[name] = test_url(name, url)

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} URLs accessible")

    if passed < total:
        print("\n⚠️  WARNING: Some URLs failed. Database downloads may not work.")
        print("Check:")
        print("  1. Internet connection")
        print("  2. Firewall/proxy settings")
        print("  3. Database provider websites for URL changes:")
        print("     - GWAS: https://www.ebi.ac.uk/gwas/docs/file-downloads")
        print("     - ClinVar: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/")
        sys.exit(1)
    else:
        print("\n✓ All URLs accessible! Downloads should work.")
        sys.exit(0)


if __name__ == '__main__':
    main()
