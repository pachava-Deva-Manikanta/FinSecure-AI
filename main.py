import uvicorn


def main():
    print("Starting FinSecure AI server on http://127.0.0.1:8000 ...")
    uvicorn.run("models.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()

