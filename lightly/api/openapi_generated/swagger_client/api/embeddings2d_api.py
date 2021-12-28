"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://openapi-generator.tech
"""


import re  # noqa: F401
import sys  # noqa: F401

from lightly.api.openapi_generated.swagger_client.api_client import ApiClient, Endpoint as _Endpoint
from lightly.api.openapi_generated.swagger_client.model_utils import (  # noqa: F401
    check_allowed_values,
    check_validations,
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types
)
from lightly.api.openapi_generated.swagger_client.model.api_error_response import ApiErrorResponse
from lightly.api.openapi_generated.swagger_client.model.create_entity_response import CreateEntityResponse
from lightly.api.openapi_generated.swagger_client.model.embedding2d_create_request import Embedding2dCreateRequest
from lightly.api.openapi_generated.swagger_client.model.embedding2d_data import Embedding2dData
from lightly.api.openapi_generated.swagger_client.model.mongo_object_id import MongoObjectID


class Embeddings2dApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client
        self.create_embeddings2d_by_embedding_id_endpoint = _Endpoint(
            settings={
                'response_type': (CreateEntityResponse,),
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d',
                'operation_id': 'create_embeddings2d_by_embedding_id',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                    'embedding_id',
                    'embedding2d_create_request',
                ],
                'required': [
                    'dataset_id',
                    'embedding_id',
                    'embedding2d_create_request',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                    'embedding_id':
                        (MongoObjectID,),
                    'embedding2d_create_request':
                        (Embedding2dCreateRequest,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                    'embedding_id': 'embeddingId',
                },
                'location_map': {
                    'dataset_id': 'path',
                    'embedding_id': 'path',
                    'embedding2d_create_request': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client
        )
        self.get_embedding2d_by_id_endpoint = _Endpoint(
            settings={
                'response_type': (Embedding2dData,),
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d/{embedding2dId}',
                'operation_id': 'get_embedding2d_by_id',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                    'embedding_id',
                    'embedding2d_id',
                ],
                'required': [
                    'dataset_id',
                    'embedding_id',
                    'embedding2d_id',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                    'embedding_id':
                        (MongoObjectID,),
                    'embedding2d_id':
                        (MongoObjectID,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                    'embedding_id': 'embeddingId',
                    'embedding2d_id': 'embedding2dId',
                },
                'location_map': {
                    'dataset_id': 'path',
                    'embedding_id': 'path',
                    'embedding2d_id': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )
        self.get_embeddings2d_by_embedding_id_endpoint = _Endpoint(
            settings={
                'response_type': ([Embedding2dData],),
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d',
                'operation_id': 'get_embeddings2d_by_embedding_id',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                    'embedding_id',
                ],
                'required': [
                    'dataset_id',
                    'embedding_id',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                    'embedding_id':
                        (MongoObjectID,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                    'embedding_id': 'embeddingId',
                },
                'location_map': {
                    'dataset_id': 'path',
                    'embedding_id': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )

    def create_embeddings2d_by_embedding_id(
        self,
        dataset_id,
        embedding_id,
        embedding2d_create_request,
        **kwargs
    ):
        """create_embeddings2d_by_embedding_id  # noqa: E501

        Create a new 2d embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.create_embeddings2d_by_embedding_id(dataset_id, embedding_id, embedding2d_create_request, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_id (MongoObjectID): ObjectId of the dataset
            embedding_id (MongoObjectID): ObjectId of the embedding
            embedding2d_create_request (Embedding2dCreateRequest):

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            CreateEntityResponse
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_id'] = \
            dataset_id
        kwargs['embedding_id'] = \
            embedding_id
        kwargs['embedding2d_create_request'] = \
            embedding2d_create_request
        return self.create_embeddings2d_by_embedding_id_endpoint.call_with_http_info(**kwargs)

    def get_embedding2d_by_id(
        self,
        dataset_id,
        embedding_id,
        embedding2d_id,
        **kwargs
    ):
        """get_embedding2d_by_id  # noqa: E501

        Get the 2d embeddings by id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_embedding2d_by_id(dataset_id, embedding_id, embedding2d_id, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_id (MongoObjectID): ObjectId of the dataset
            embedding_id (MongoObjectID): ObjectId of the embedding
            embedding2d_id (MongoObjectID): ObjectId of the 2d embedding

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            Embedding2dData
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_id'] = \
            dataset_id
        kwargs['embedding_id'] = \
            embedding_id
        kwargs['embedding2d_id'] = \
            embedding2d_id
        return self.get_embedding2d_by_id_endpoint.call_with_http_info(**kwargs)

    def get_embeddings2d_by_embedding_id(
        self,
        dataset_id,
        embedding_id,
        **kwargs
    ):
        """get_embeddings2d_by_embedding_id  # noqa: E501

        Get all 2d embeddings of an embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_embeddings2d_by_embedding_id(dataset_id, embedding_id, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_id (MongoObjectID): ObjectId of the dataset
            embedding_id (MongoObjectID): ObjectId of the embedding

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            [Embedding2dData]
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_id'] = \
            dataset_id
        kwargs['embedding_id'] = \
            embedding_id
        return self.get_embeddings2d_by_embedding_id_endpoint.call_with_http_info(**kwargs)

