import psycopg2
import psycopg2.extras
import db_params
import tensorflow_hub as hub
import numpy as np
import json
import re
import requests
import math
import tensorflow_text
from pprint import pprint
from datetime import datetime, timedelta
import pandas as pd
import pymorphy2

from natasha import (
	Segmenter,
	MorphVocab,

	NewsEmbedding,
	NewsMorphTagger,
	NewsSyntaxParser,
	NewsNERTagger,

	PER,
	NamesExtractor,

	Doc
)


class LukoshkoTextToEmbeddings:
	def __init__(self, db_params, table_name, table_name_ner, table_similar_articles):
		self._create_table(db_params, table_name)
		self._create_table_ner(db_params, table_name_ner)
		self._create_table_similar_articles(db_params, table_similar_articles)
		# self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

		self.segmenter = Segmenter()
		self.morph_vocab = MorphVocab()

		emb = NewsEmbedding()
		self.morph_tagger = NewsMorphTagger(emb)
		self.syntax_parser = NewsSyntaxParser(emb)
		self.ner_tagger = NewsNERTagger(emb)
		self.names_extractor = NamesExtractor(self.morph_vocab)
		self.morph = pymorphy2.MorphAnalyzer()

	def _get_normal_PER_spans(self, article):
		doc = Doc(article)

		doc.segment(self.segmenter)

		doc.tag_morph(self.morph_tagger)

		for token in doc.tokens:
			token.lemmatize(self.morph_vocab)

		doc.tag_ner(self.ner_tagger)

		for span in doc.spans:
			span.normalize(self.morph_vocab)

		for span in doc.spans:
			if span.type == PER:
				span.extract_fact(self.names_extractor)

		return doc.spans

	def get_nominative(self, name):
		'''
		Pass only a single word as parameter! Example: "Вася" (not Вася Пупкин!!!)
		'''

		# костыль
		if name.strip()[-2:] in 'ов ев ий ва ян':
			return name

		if name.lower() in 'кызы улуу':
			return name

		morth_res = self.morph.parse(name)

		max_index = 0
		max_score = 0
		index = 0

		for morph_info in morth_res:

			if 'NOUN' in morph_info.tag and 'plur' not in morph_info.tag:
				if 'nomn' in morph_info.tag:
					max_index = index
					max_score = morph_info.score + 1

				elif morph_info.tag.gender == 'masc':
					if morph_info.score > max_score:
						max_index = index
						max_score = morph_info.score

				elif morph_info.tag.gender == 'femn':
					if morph_info.score > max_score:
						max_index = index
						max_score = morph_info.score


				elif 'sing' in morph_info.tag:
					if morph_info.score > max_score:
						max_index = index
						max_score = morph_info.score

				else:
					return name

			index += 1

		if morth_res[max_index].tag.gender == 'femn':
			return morth_res[max_index].inflect({'nomn', 'femn'}).word.title()
		else:
			return morth_res[max_index].normal_form.title()

	def generate_NER(self, article):
		list_of_PERS = []

		spans = self._get_normal_PER_spans(article)

		for span in spans:

			info = {}

			if span.type == 'PER':
				info['PER'] = span.text
				info['PER_normal'] = span.normal

				list_of_PERS.append(info)

		df_PERS = pd.DataFrame(list_of_PERS)

		# add nominative column
		for i, row in df_PERS.iterrows():
			NOM_name = ' '.join([self.get_nominative(x) for x in row['PER_normal'].split(' ')])
			df_PERS.at[i, 'NOM_name'] = NOM_name

		# remove (combine same NERs)
		for i, row in df_PERS.iterrows():
			filter_df = df_PERS[df_PERS['NOM_name'].str.contains(row['NOM_name'], regex=False)]
			if filter_df.shape[0] > 1:
				df_PERS.drop(i, inplace=True)

		return df_PERS

	def find_similiar_articles(self, article, sentence_limit=None, distance_cutoff=0.99, overlap=2):
		"""
		Find similiar articles

		:param article: Article text to find
		:param sentence_limit: Sentence limit for article
		:param distance_cutoff: Cutoff for distance (the lower the more accurate)
		:param overlap: Overlaps quantity
		:return: returns pandas dataframe
		"""

		sentences = re.split(r'(?<=[^А-Я].[.? \n]) +(?=[А-Я \n])', article)

		if sentence_limit:
			sentences = sentences[:sentence_limit]

		sentence_distances = self.get_distances(sentences)

		list_of_data = []

		for distance_data in sentence_distances:
			list_of_indexes = []
			for index in distance_data['metadata']:
				list_of_indexes.append(int(index))

			# get additional metadata info
			dist_metadata = self.get_metadata(list_of_indexes)

			for metadata in dist_metadata:
				info = {
					'distance': distance_data['metadata'][str(metadata['faiss_index'])]['distance'],
					'mass_media_name': metadata['mass_media_name'],
					'sentence_number': metadata['sentence_number'],
					'title': metadata['title'],
					'sentence': metadata['sentence'],
					'sentence_hash': metadata['sentence_hash'],
					'lang': metadata['lang'],
					'scraping_time': metadata['scraping_time'],
					'url': metadata['url'],
					'mass_media_maintext_hash': metadata['mass_media_maintext_hash'],
					'clean_sentence': metadata['clean_sentence'],
					# 'serialized_array_meta': metadata['serialized_array_meta'],
					# 'array_bytes': metadata['array_bytes'],
					'scraper_idx': metadata['scraper_idx'],
					'faiss_index': metadata['faiss_index']
				}

				list_of_data.append(info)

		df_distances = pd.DataFrame(list_of_data)

		# distance_cutoff
		df_distances = df_distances[df_distances['distance'] <= distance_cutoff]

		# overlaps
		df_overlaps = df_distances['mass_media_maintext_hash'].value_counts().to_frame().reset_index()
		df_overlaps.rename(columns={'index': 'article_hash', 'mass_media_maintext_hash': 'overlaps'}, inplace=True)

		df_overlaps = df_overlaps[df_overlaps['overlaps'] >= overlap]

		# mass media info
		df_mass_media = df_distances[['mass_media_name', 'mass_media_maintext_hash', 'title', 'url']].drop_duplicates(
			keep='first')

		df_overlaps = pd.merge(df_overlaps, df_mass_media, how='left', left_on='article_hash',
							   right_on='mass_media_maintext_hash')

		return df_overlaps

	def get_distances(self, sentences):
		response = requests.post(
			'https://lukoshkoapi.kloop.io/api/v1/text_range_search/',
			json={
				"sentences": sentences,
				"db": "scrapers",
				"table": "media",
				"with_meta": "False",
				"meta_with_embeddings": "False",
				"top_k": 10
			})

		result = json.loads(response.content.decode())

		return result

	def get_metadata(self, list_of_indexes):
		response = requests.post(
			'https://lukoshkoapi.kloop.io/api/v1/text_search/',
			json={
				"text": list_of_indexes,
				"db": "scrapers",
				"table": "media",
				"search_column": "faiss_index",
				"is_array": "True"
			})

		result = json.loads(response.content.decode())

		return result

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
					f'scraping_time TIMESTAMP,'
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

	def _create_table_ner(self, db_params, table_name):
		conn = None
		try:
			conn = psycopg2.connect(**db_params)
			with conn.cursor() as cursor:
				cursor.execute(
					f'CREATE TABLE IF NOT EXISTS {table_name}('
					f'    url             TEXT   NOT NULL,'
					f'    nom_name        TEXT,'
					f'    faiss_index     SERIAL NOT NULL,'
					f'    CONSTRAINT {table_name}_url_nom_pk'
					f'        PRIMARY KEY (url, nom_name));'
				)
				cursor.execute(
					f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_url_nom_uindex '
					f'ON mass_media_ner (url, nom_name);'
				)
				conn.commit()
		except Exception as e:
			print(e)
		finally:
			if conn:
				conn.close()

	def _create_table_similar_articles(self, db_params, table_name):
		conn = None
		try:
			conn = psycopg2.connect(**db_params)
			with conn.cursor() as cursor:
				cursor.execute(
					f'CREATE TABLE IF NOT EXISTS {table_name}'
					f'('
					f'    origin_mass_media_name       TEXT,'
					f'    origin_title                 TEXT,'
					f'    origin_author                TEXT,'
					f'    origin_date_publish          TEXT,'
					f'    origin_url                   TEXT,'
					f'    origin_tags                  TEXT,'
					f'    origin_lang                  TEXT,'
					f'    origin_maintext_hash         TEXT,'
					f'    origin_scraping_time         TIMESTAMP,'
					f'    mass_media_idx               BIGINT NOT NULL,'
					f'    count_similar_sentences      INT,'
					f'    sim_mass_meida_name          TEXT,'
					f'    sim_mass_meida_maintext_hash TEXT,'
					f'    sim_title                    TEXT,'
					f'    sim_url                      TEXT,'
					f'    faiss_index                  SERIAL NOT NULL,'
					f'    CONSTRAINT {table_name}_pk PRIMARY KEY'
					f'        (origin_url, origin_maintext_hash, sim_mass_meida_maintext_hash, sim_url),'
					f'    CONSTRAINT {table_name}_url_hash_fk foreign key (origin_url, origin_maintext_hash)'
					f'        REFERENCES mass_media'
					f');'
				)
				cursor.execute(
					f'CREATE UNIQUE INDEX IF NOT EXISTS {table_name}_urls_hashes_uindex'
					f'    ON {table_name} (origin_url, sim_url, origin_maintext_hash, sim_mass_meida_maintext_hash);'
				)
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

	def _insert_ner(self, conn, rows, table_name):
		try:
			with conn.cursor() as cursor:
				psycopg2.extras.execute_batch(cursor,
											  f"""
	                                    INSERT INTO {table_name}(
	                                         url,
											 nom_name
	                                     )
										VALUES (
											%(url)s,
											%(nom_name)s
										)
										ON CONFLICT ON CONSTRAINT {table_name}_url_nom_pk DO NOTHING;""",
											  rows, page_size=1000)
			conn.commit()
		except Exception as e:
			print(e)

	def _insert_similar_article(self, conn, rows, table_name):
		try:
			with conn.cursor() as cursor:
				psycopg2.extras.execute_batch(cursor,
											  f"""
	                                    INSERT INTO {table_name}(
	                                         origin_mass_media_name,
	                                         origin_title,
	                                         origin_author,
	                                         origin_date_publish,
	                                         origin_url,
	                                         origin_tags,
	                                         origin_lang,
	                                         origin_maintext_hash,
	                                         origin_scraping_time,
	                                         mass_media_idx,
	                                         count_similar_sentences,
	                                         sim_mass_meida_name,
	                                         sim_mass_meida_maintext_hash,
	                                         sim_title,
	                                         sim_url
	                                     )
										VALUES (
											%(mass_media_name)s,
											%(title)s,
											%(author)s,
											%(date_publish)s,
											%(url)s,
											%(tags)s,
											%(lang)s,
											%(maintext_hash)s,
											%(scraping_time)s,
											%(mass_media_idx)s,
											%(count_similar_sentences)s,
											%(sim_mass_media_name)s,
											%(sim_mass_media_maintext_hash)s,
											%(sim_title)s,
											%(sim_url)s
										)
										ON CONFLICT ON CONSTRAINT {table_name}_pk DO NOTHING;""",
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
				print("insert embeddings", len(rows))

		except (Exception, psycopg2.DatabaseError) as error:
			print(error)
		finally:
			if conn:
				conn.close()
			if insert_conn:
				insert_conn.close()

	def create_NERs(self, db_params, source_table_name, target_table_name):
		conn = None
		insert_conn = None

		try:
			conn = psycopg2.connect(**db_params)
			insert_conn = psycopg2.connect(**db_params)
			rows = []
			with conn.cursor('server_side_cursor', cursor_factory=psycopg2.extras.DictCursor) as cursor:
				today = datetime.now().strftime("%Y-%m-%d")
				cursor.execute(f"""
								SELECT *
								FROM {source_table_name}
								WHERE url in (SELECT s.url
											  FROM {source_table_name} s
											  where s.lang = 'ru'
												and mass_media_name != 'mk' EXCEPT
											  SELECT tg.url
											  FROM {target_table_name} tg)
								  and scraping_time
									between to_timestamp('2021-07-01', 'YYYY-MM-DD') and to_timestamp('{today}', 'YYYY-MM-DD')
								  and mass_media_name != 'mk';
				""")
				for i, row in enumerate(cursor):
					maintext = row['maintext']
					ner = self.generate_NER(maintext)
					for i, irow in ner.iterrows():
						data = {
							'url': row['url'],
							'nom_name': irow["NOM_name"]
						}

						rows.append(data)

					if len(rows) % 1000 == 0:
						self._insert_ner(insert_conn, rows, target_table_name)
						rows = []

				self._insert_ner(insert_conn, rows, target_table_name)
				print("Insert NER", len(rows))
		except (Exception, psycopg2.DatabaseError) as error:
			print(error)
		finally:
			if conn:
				conn.close()
			if insert_conn:
				insert_conn.close()

	def create_similar_articles(self, db_params, source_table_name, target_table_name):
		conn = None
		insert_conn = None

		try:
			conn = psycopg2.connect(**db_params)
			insert_conn = psycopg2.connect(**db_params)
			rows = []
			with conn.cursor('server_side_cursor', cursor_factory=psycopg2.extras.DictCursor) as cursor:
				cursor.execute(f"""
								SELECT *
								FROM {source_table_name}
								WHERE mass_media_idx in (SELECT s.mass_media_idx
														 FROM {source_table_name} s
														 where s.lang = 'ru'
														   and mass_media_name != 'mk' EXCEPT
														 SELECT tg.mass_media_idx
														 FROM {target_table_name} tg
														 where tg.origin_lang = 'ru')
								  and scraping_time
									between TO_TIMESTAMP('2021-08-01', 'YYYY-MM-DD') and TO_TIMESTAMP('2021-08-11', 'YYYY-MM-DD')
								  and mass_media_name != 'mk';
				""")
				for i, row in enumerate(cursor):
					columns = list(row.keys())
					maintext = row['maintext']
					df_similar_articles = self.find_similiar_articles(maintext)
					print(f"article#{i}")

					for j, irow in df_similar_articles.iterrows():
						data = dict(zip(columns, row))
						data.update({
							'count_similar_sentences': int(irow['overlaps']),
							'sim_mass_media_name': irow['mass_media_name'],
							'sim_mass_media_maintext_hash': irow['mass_media_maintext_hash'],
							'sim_title': irow['title'],
							'sim_url': irow['url']
						})
						rows.append(data)

						if len(rows) % 1000 == 0:
							self._insert_similar_article(insert_conn, rows, target_table_name)
							rows = []

				self._insert_similar_article(insert_conn, rows, target_table_name)
				print("Insert similar articles")
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

	def trigger_mass_media_text_create_index(self, token):
		data = {
			"table_name": "mass_media_sentence_with_embeddings",
			"source_db_name": "scrapers",
			"alias": "media",
			"unique_columns": ["url", "sentence_hash"],
			"column_to_embed": "sentence",
			"add_directly": "True"
		}
		response = requests.post(
			"https://lukoshkoapi.kloop.io/api/v1/text_create_index/",
			data=data,
			headers={
				"Authorization": f"Token {token}"
			})
		print(response.content)

if __name__ == "__main__":
	start = datetime.now()
	print(f'start: {start}')
	lukoshko_text = LukoshkoTextToEmbeddings(db_params.db_params,
											 db_params.table_name_posts,
											 'mass_media_ner',
											 'mass_media_similar_articles')
	# lukoshko_text.create_embeddings(db_params.db_params,
	# 								'mass_media_sentence',
	# 								db_params.table_name_posts)
	lukoshko_text.create_NERs(db_params.db_params,
							  'mass_media',
							  'mass_media_ner')
	# lukoshko_text.trigger_mass_media_text_create_index(db_params.token)
	# lukoshko_text.create_similar_articles(db_params.db_params,
	# 									  'mass_media',
	# 									  'mass_media_similar_articles')
	end = datetime.now()
	print(f'end: {end}\nexecution time: {end - start}')
