from flask import Flask, request
import sqlite3

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/api/v1/resources/books/all', methods=['GET'])
def get_all():
    connection = sqlite3.connect('books.db')
    cursor = connection.cursor()
    select_books = "SELECT * FROM books"
    result = cursor.execute(select_books).fetchall()
    connection.close()
    return {'books': result}


@app.route('/api/v1/resources/book/<string:author>', methods=['GET'])
def get_by_author(author):
    connection = sqlite3.connect('books.db')
    cursor = connection.cursor()
    select_books = "SELECT * FROM books WHERE author=?"
    result = cursor.execute(select_books, (author,)).fetchall()
    connection.close()
    return {'books': result}


@app.route('/api/v1/resources/book/filter', methods=['GET'])
def filter_table():
    query_parameters = request.get_json()

    book_id = query_parameters.get('id')
    published = query_parameters.get('published')
    author = query_parameters.get('author')

    connection = sqlite3.connect('books.db')
    cursor = connection.cursor()

    query = "SELECT * FROM books WHERE"
    to_filter = []

    if book_id:
        query += ' id=? AND'
        to_filter.append(book_id)
    if published:
        query += ' published=? AND'
        to_filter.append(published)
    if author:
        query += ' author=? AND'
        to_filter.append(author)

    if not (book_id or published or author):
        return "Page not found 404"

    query = query[:-4] + ';'
    result = cursor.execute(query, to_filter).fetchall()
    connection.close()
    return {'books': result}


if __name__ == "__main__":
    app.run()


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8002, debug=True)