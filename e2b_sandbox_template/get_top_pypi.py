import requests

def get_top_pypi_packages(limit=50):
    # API endpoint for PyPI download statistics (30 days)
    url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Get the top packages
        top_packages = data["rows"][:limit]
        return [pkg['project'] for pkg in top_packages]
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def write_to_file(packages, filename="top_pypi_packages.txt"):
    try:
        with open(filename, "w") as file:
            for i, package in enumerate(packages, 1):
                file.write(f"{package}\n")
        print(f"Successfully wrote top {len(packages)} packages to {filename}")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    top_packages = get_top_pypi_packages(100)
    if top_packages:
        write_to_file(top_packages) 