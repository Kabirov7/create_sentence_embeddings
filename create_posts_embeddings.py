import psycopg2
import psycopg2.extras
import db_params
import re
import tensorflow_hub as hub
import json
import numpy as np
import math
import tensorflow_text
from pprint import pprint
from datetime import datetime


class LukoshkoTextToEmbeddings:
	def __init__(self, db_params, table_name):
		self._create_table(db_params, table_name)
		self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

	def _create_table(self, db_params, table_name):
		conn = None
		try:
			conn = psycopg2.connect(**db_params)
			with conn.cursor() as cursor:
				cursor.execute(
					f'create table if not exists {table_name}('
					f'mass_media_name text,'
					f'sentence_number int,'
					f'title text,'
					f'sentence text,'
					f'sentence_hash text,'
					f'lang text,'
					f'scraping_time text,'
					f'url text,'
					f'mass_media_maintext_hash text,'
					f'clean_sentence text,'
					f'serialized_array_meta text,'
					f'array_bytes bytea,'  # scraper_idx
					f'scraper_idx bigint not null,'
					f'faiss_index serial not null,'
					f'UNIQUE (url, sentence_hash))')
				conn.commit()
		except Exception as e:
			print(e)
		finally:
			if conn:
				conn.close()

	def _insert_sentence(self, conn, rows, table_name):
		try:
			with conn.cursor() as cursor:
				psycopg2.extras.execute_batch(cursor,
											  f"""
                                    INSERT INTO {table_name}(
                                         mass_media_name,
                                         sentence_number,
                                         title,
                                         sentence,
                                         sentence_hash,
                                         lang,
                                         scraping_time,
                                         url,
                                         mass_media_maintext_hash,
                                         clean_sentence,
                                         serialized_array_meta,
                                         array_bytes,
										 scraper_idx
                                     )
									VALUES (
										%(mass_media_name)s,
										%(sentence_number)s,
										%(title)s,
										%(sentence)s,
										%(sentence_hash)s,
										%(lang)s,
										%(scraping_time)s,
										%(url)s,
										%(mass_media_maintext_hash)s,
										%(clean_sentence)s,
										%(serialized_array_meta)s,
										%(array_bytes)s,
										%(scraper_idx)s
									)
									ON CONFLICT ON CONSTRAINT mass_media_sentence_with_embeddings_url_sentence_hash_key DO NOTHING;""",
											  rows, page_size=1000)
			conn.commit()
		except Exception as e:
			print(e)

	def create_embeddings(self, db_params, source_table_name, target_table_name):
		conn = None
		insert_conn = None
		
		try:
			conn = psycopg2.connect(**db_params)
			insert_conn = psycopg2.connect(**db_params)
			rows = []
			with conn.cursor('server_side_cursor', cursor_factory=psycopg2.extras.DictCursor) as cursor:
				cursor.execute(f"""
						SELECT * FROM {source_table_name}
							WHERE scraper_idx in (SELECT s.scraper_idx
												  FROM {source_table_name} s where s.lang = 'ru' EXCEPT
												  SELECT tg.scraper_idx
												  FROM {target_table_name} tg where tg.lang = 'ru');""")
				for i, row in enumerate(cursor):
					columns = list(row.keys())
					sentence = row['sentence']
					clean_sentence = self._clean_sentence(sentence)
					embed_sentence, embed_meta = self._get_embedding_bytes(clean_sentence)
					data = dict(zip(columns, row))
					data.update({
						'sentence': sentence,
						'clean_sentence': clean_sentence,
						'serialized_array_meta': embed_meta,
						'array_bytes': embed_sentence
					})

					rows.append(data)
					i += 1
					if len(rows) % 1000 == 0:
						self._insert_sentence(insert_conn, rows, target_table_name)
						rows = []

				self._insert_sentence(insert_conn, rows, target_table_name)

		except (Exception, psycopg2.DatabaseError) as error:
			print(error)
		finally:
			if conn:
				conn.close()
			if insert_conn:
				insert_conn.close()

	def _get_sentences(self, comment):
		comment = re.sub(r'[.?!]', '\g<0> ', comment).replace('\n', '.')
		sentences = re.split(r'(?<=[^А-Я].[.?!])[ \s]*(?=[А-Яа-я])', comment)
		return sentences

	def _clean_sentence(self, sentence):
		return re.compile(r"\s+").sub(" ", "".join(re.findall(r'(\w+|[!?.,:;\- ])', str(sentence))))

	def _get_embedding_bytes(self, sentence):
		if sentence:
			embedding = np.array(self.embed(sentence.strip().lower()))
			array_meta = json.dumps({'array_meta':
										 {'dtype': embedding.dtype.str,
										  'shape': embedding.shape}})
			return embedding.tobytes(), array_meta
		else:
			return None, None


if __name__ == "__main__":
	lukoshko_text = LukoshkoTextToEmbeddings(db_params.db_params,
											 db_params.table_name_posts)
	lukoshko_text.create_embeddings(db_params.db_params,
									'mass_media_sentence',
									db_params.table_name_posts)
