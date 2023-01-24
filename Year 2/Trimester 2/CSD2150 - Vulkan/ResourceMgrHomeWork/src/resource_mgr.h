/*!*****************************************************************************
\file   resource_mgr.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: CSD2150
\par Section: A
\par Assignment 1
\date 24th Janunary 2023
\brief
This file contains function definition for a resource manager
*******************************************************************************/
#include "xcore/src/xcore.h"

// This is an example of a resource system
namespace resource
{
    struct mgr;

    // The actual GUID
    // All resources GUIDs has the very first bit turn on
    // This bit servers as a flag to know if a reference is an actual GUID or a pointer
    using guid      = xcore::guid::unit<64, struct resource_guid_tag>;
    /*!*****************************************************************************
        \brief Creates and return a unique guid
    *******************************************************************************/
    static inline guid CreateUniqueGuid( void ) noexcept 
    { 
        guid id{};
        id.Reset();
        id.m_Value |= 0b1;
        return id;
    }

    // This is the type of the resource described as a simple GUID
    using type_guid = xcore::guid::unit<64, struct resource_type_guid_tag>;

    // Resource type
    template< typename T_RESOURCE_STRUCT_TYPE >
    struct type
    {
        // Expected static parameters
         constexpr static inline auto                 name_v = "Custom Resource Name";  // **** This is opcional *****
         constexpr static inline resource::type_guid  guid_v = { "ResourceName" };        
         static T_RESOURCE_STRUCT_TYPE*  Load   (                               resource::mgr& Mgr, resource::guid GUID );
         static void                     Destroy( T_RESOURCE_STRUCT_TYPE& Data, resource::mgr& Mgr, resource::guid GUID );
    };

    template< typename T, typename = void > 
    struct get_custom_name                                               
    { 
        static inline const char* value = []
        {
            return typeid(T).name();
        } ();  
    };
    template< typename T >                  
    struct get_custom_name< T, std::void_t< typename type<T>::name_v > > 
    { 
        static inline constexpr const char* value = type<T>::name_v;   
    };

    // this simple structure has the actual reference to the resource
    // but it needs to know if the actual reference is a pointer or
    // is a GUID
    union partial_ref
    {
        guid    m_GUID;                 // This is 64 bits
        void*   m_Pointer;              // This is 64 bits

        constexpr bool isPointer() const 
        { 
            /*TODO*/ 
            return (m_GUID.m_Value & 0b1) == 0;
        }
    };

    // This structure is able to hold a reference to any kind of resource
    // This is also a full reference rather than partial
    struct universal_ref
    {
        partial_ref   m_PRef;
        type_guid     m_TypeGUID;

        // No copy allowed... 
                                universal_ref   ()                     = default;
                                universal_ref   (const universal_ref&) = delete;
        const universal_ref&    operator =      (const universal_ref&) = delete;

        /*!*****************************************************************************
            \brief Move constructor
        *******************************************************************************/
        universal_ref(universal_ref&& A)
        {
            m_PRef = A.m_PRef;
            m_TypeGUID = A.m_TypeGUID;
            A.m_PRef.m_GUID.m_Value = A.m_TypeGUID.m_Value = 0;
        }


        /*!*****************************************************************************
            \brief Move assignment operator
        *******************************************************************************/
        universal_ref& operator = (universal_ref&& A)
        {
            m_PRef = A.m_PRef;
            m_TypeGUID = A.m_TypeGUID;
            A.m_PRef.m_GUID.m_Value = A.m_TypeGUID.m_Value = 0;
            return *this;
        }

        // Destruction can not happen carelessly 
        ~universal_ref()
        {
            // No references can die with pointers they must be kill by the user properly 
            // Since we need to update the ref counts
            assert(m_PRef.isPointer() == false || m_PRef.m_GUID.m_Value == 0);
        }
    };

    // This structure is able to hold a reference to a particular resource type
    // The type for T_TYPE_V could be:
    // resource::type<texture>, resource::type<sound>, etc...
    // This structure is also a full reference rather than partial
    template < typename T_RESOURCE_TYPE >
    struct ref
    {
        using                           type = T_RESOURCE_TYPE;
        partial_ref                     m_PRef;
        static inline constexpr auto&   m_TypeGUID = resource::type<T_RESOURCE_TYPE>::guid_v;

        // No copy allowed... 
                                        ref             () = default;
                                        ref             (const ref&) = delete;
        const ref&                      operator =      (const ref&) = delete;

        /*!*****************************************************************************
            \brief Move Constructor
        *******************************************************************************/
        ref(ref&& A)
        {
            /*TODO*/
            m_PRef = A.m_PRef;
            A.m_PRef.m_GUID.m_Value = 0;
        }

        /*!*****************************************************************************
            \brief Move assignment operator
        *******************************************************************************/
        ref& operator = (ref&& A)
        {
            /*TODO*/
            m_PRef = A.m_PRef;
            A.m_PRef.m_GUID.m_Value = 0;
            return *this;
        }

        // Destruction can not happen carelessly 
        ~ref()
        {
            // No references can die with pointers they must be kill by the user properly 
            // Since we need to update the ref counts
            assert( m_PRef.isPointer() == false || m_PRef.m_GUID.m_Value == 0 );
        }
    };

    //
    // RSC MANAGER
    //
    namespace details
    {
        struct instance_info
        {
            void*           m_pData     { nullptr };
            resource::guid  m_GUID;
            type_guid       m_TypeGUID;
            int             m_RefCount  { 1 };
        };

        struct universal_type
        {
            using load_fun      = void* ( resource::mgr& Mgr, guid GUID );
            using destroy_fun   = void  ( void* pData, resource::mgr& Mgr, resource::guid GUID );

            type_guid       m_GUID;
            load_fun*       m_pLoadFunc;
            destroy_fun*    m_pDestroyFunc;
            const char*     m_pName;
        };
    }

    // Resource Manager
    struct mgr
    {
        /*!*****************************************************************************
            \brief Constructor for mgr
        *******************************************************************************/
        mgr()
        {
            m_pInfoBufferEmptyHead = &m_InfoBuffer[0];
            for (size_t i{ 1 }; i < max_resources_v; ++i)
            {
                m_pInfoBufferEmptyHead->m_pData = &m_InfoBuffer[i];
                m_pInfoBufferEmptyHead = reinterpret_cast<details::instance_info*>(m_pInfoBufferEmptyHead->m_pData);
            }

            m_pInfoBufferEmptyHead->m_pData = nullptr;
            m_pInfoBufferEmptyHead = &m_InfoBuffer[0];
        }

        /*!*****************************************************************************
            \brief Based on the variadic template arguments, add the types into 
                   m_RegisteredTypes map
        *******************************************************************************/
        template< typename...T_ARGS >
        void RegisterTypes()
        {
            //
            // Insert all the types into the hash table
            //
            (   [&]< typename T >(T*)
                {
                    m_RegisteredTypes.emplace
                    ( type<T>::guid_v.m_Value
                    , details::universal_type
                        { {type<T>::guid_v}
                        , [](resource::mgr& Mgr, guid GUID) constexpr -> void*
                            { return type<T>::Load(Mgr,GUID); }
                        , [](void* pData, resource::mgr& Mgr, guid GUID) constexpr
                            { type<T>::Destroy(*reinterpret_cast<T*>(pData),Mgr,GUID); }
                        , get_custom_name<T>::value
                        }
                    );
                }(reinterpret_cast<T_ARGS*>(0))
                , ...
            );
        }

        /*!*****************************************************************************
            \brief Get the resource of type T, if an instance of it is not loaded, this
                   function will load it into the mgr

            \param [in, out] R: A reference to resource type

            \return A pointer to the resource type
        *******************************************************************************/
        template< typename T >
        T* getResource( resource::ref<T>& R )
        {
            if (R.m_PRef.isPointer())
                return reinterpret_cast<T*>(R.m_PRef.m_Pointer);

            uint64_t hashID = R.m_PRef.m_GUID.m_Value ^ R.m_TypeGUID.m_Value;

            if (auto entry = m_ResourceInstance.find(hashID); entry != m_ResourceInstance.end())
            {
                entry->second->m_RefCount++;
                R.m_PRef.m_Pointer = entry->second->m_pData;
                return reinterpret_cast<T*>(R.m_PRef.m_Pointer);
            }
            T* pRsc = type<T>::Load(*this, R.m_PRef.m_GUID);
            // If the user return nulls it must mean that it failed to load so we could return a
            // temporary resource of the right type
            if (!pRsc) return nullptr;

            FullInstanceInfoAlloc(pRsc, R.m_PRef.m_GUID, R.m_TypeGUID);
            return reinterpret_cast<T*>(R.m_PRef.m_Pointer = pRsc);
        }

        /*!*****************************************************************************
            \brief Get the resource of universal type. If an instance of this universal 
                   reference is not loaded, this functill will load it into the mgr

            \param [in, out] URef: A universal reference to the resource

            \return A pointer to the resource type
        *******************************************************************************/
        void* getResource( universal_ref& URef )
        {
            if (URef.m_PRef.isPointer())
                return reinterpret_cast<void*>(URef.m_PRef.m_Pointer);

            uint64_t hashID = URef.m_PRef.m_GUID.m_Value ^ URef.m_TypeGUID.m_Value;

            if (auto entry = m_ResourceInstance.find(hashID); entry != m_ResourceInstance.end())
            {
                ++entry->second->m_RefCount;
                URef.m_PRef.m_Pointer = entry->second->m_pData;
                return reinterpret_cast<void*>(URef.m_PRef.m_Pointer);
            }

            auto const& universalType = m_RegisteredTypes.find(URef.m_TypeGUID.m_Value);
            assert(universalType != m_RegisteredTypes.end()); // Type was not registered

            void* pRsc = universalType->second.m_pLoadFunc(*this, URef.m_PRef.m_GUID);
            // If the user return nulls it must mean that it failed to load so we could return a
            // temporary resource of the right type
            if (!pRsc) return nullptr;

            FullInstanceInfoAlloc(pRsc, URef.m_PRef.m_GUID, URef.m_TypeGUID);
            return reinterpret_cast<void*>(URef.m_PRef.m_Pointer = pRsc);
        }

        /*!*****************************************************************************
            \brief  Release the resource of type T from mgr

            \param [in, out] Ref: Resource of type T to be removed from the manager
        *******************************************************************************/
        template< typename T >
        void ReleaseRef(resource::ref<T>& Ref )
        {
            uint64_t const instanceKey = Ref.m_PRef.m_GUID.m_Value ^ Ref.m_TypeGUID.m_Value, releaseKey = Ref.m_PRef.m_GUID.m_Value;
            auto const& instance = m_ResourceInstance.find(instanceKey), 
                        release  = m_ResourceInstanceRelease.find(releaseKey);

            Ref.m_PRef.m_GUID.m_Value |= 0b1;

            if (release != m_ResourceInstanceRelease.end())
            {
                resource::guid const tmp = release->second->m_GUID;
                if (0 >= --release->second->m_RefCount)
                    FullInstanceInfoRelease(*release->second);
                Ref.m_PRef.m_GUID = tmp;
            }
            else if (instance != m_ResourceInstance.end())
            {
                details::instance_info* const tmp = reinterpret_cast<details::instance_info*>(instance->second);
                if (0 >= --instance->second->m_RefCount)
                    FullInstanceInfoRelease(*tmp);
                Ref.m_PRef.m_GUID = tmp->m_GUID;
            }
        }

        /*!*****************************************************************************
            \brief Release the universal ref of the resource

            \param [in, out] URef: Resource of the universal reference to be removed
                   from the manager
        *******************************************************************************/
        void ReleaseRef( universal_ref& URef )
        {
            uint64_t const instanceKey = URef.m_PRef.m_GUID.m_Value ^ URef.m_TypeGUID.m_Value, releaseKey = URef.m_PRef.m_GUID.m_Value;
            auto const& instance = m_ResourceInstance.find(instanceKey),
                        release  = m_ResourceInstanceRelease.find(releaseKey);

            URef.m_PRef.m_GUID.m_Value |= 0b1;

            if (release != m_ResourceInstanceRelease.end())
            {
                resource::guid const tmp = release->second->m_GUID;
                if (0 >= --release->second->m_RefCount)
                    FullInstanceInfoRelease(*release->second);
                URef.m_PRef.m_GUID = tmp;
            }
            else if (instance != m_ResourceInstance.end())
            {
                details::instance_info* const tmp = reinterpret_cast<details::instance_info*>(instance->second);
                if (0 >= --instance->second->m_RefCount)
                    FullInstanceInfoRelease(*tmp);
                URef.m_PRef.m_GUID = tmp->m_GUID;
            }
        }

        /*!*****************************************************************************
            \brief Get the guid of resource type T

            \param [in] R: To find the guid of this type based on it's pointer

            \return the guid tagged to this R
        *******************************************************************************/
        template< typename T >
        guid getInstanceGuid( const resource::ref<T>& R ) const
        {
            auto const& id = m_ResourceInstanceRelease.find( reinterpret_cast<uint64_t>( R.m_PRef.m_Pointer ) );
            if (id != m_ResourceInstanceRelease.end())
                return id->second->m_GUID;
            return R.m_PRef.m_GUID;
        }

        /*!*****************************************************************************
            \brief Get the guid of this universal reference type

            \param [in] URef: To find the guid of this universal reference type based 
                   it's pointer

            \return the guid tagged to this URef
        *******************************************************************************/
        guid getInstanceGuid( const universal_ref& URef ) const
        {
            auto const& id = m_ResourceInstanceRelease.find(reinterpret_cast<uint64_t>(URef.m_PRef.m_Pointer));
            if (id != m_ResourceInstanceRelease.end())
                return id->second->m_GUID;
            return URef.m_PRef.m_GUID;
        }

        /*!*****************************************************************************
            \brief Make a clone of Ref into the Dest

            \param [in, out] Dest: To clone the data from Ref into Dest
            \param [in] Ref: The data to be cloned
        *******************************************************************************/
        template< typename T >
        void CloneRef( ref<T>& Dest, const ref<T>& Ref ) noexcept
        {
            ReleaseRef(Dest);
            uint64_t const instanceKey = Ref.m_PRef.m_GUID.m_Value ^ Ref.m_TypeGUID.m_Value, releaseKey = Ref.m_PRef.m_GUID.m_Value;
            auto const& instance = m_ResourceInstance.find(instanceKey),
                        release  = m_ResourceInstanceRelease.find(releaseKey);

            if (release != m_ResourceInstanceRelease.end())
            {
                Dest.m_PRef.m_Pointer = Ref.m_PRef.m_Pointer;
                ++release->second->m_RefCount;
            }
            else if (instance != m_ResourceInstance.end())
                Dest.m_PRef.m_Pointer = instance->second->m_pData;
        }

        /*!*****************************************************************************
            \brief Make a clone of Ref into Dest

            \param [in, out] Dest: To clone the data from Ref into Dest
            \param [in] Ref: The data to be cloned
        *******************************************************************************/
        void CloneRef(universal_ref& Dest, const universal_ref& URef ) noexcept
        {
            ReleaseRef(Dest);
            uint64_t const instanceKey = URef.m_PRef.m_GUID.m_Value ^ URef.m_TypeGUID.m_Value, releaseKey = URef.m_PRef.m_GUID.m_Value;
            auto const& instance = m_ResourceInstance.find(instanceKey),
                        release  = m_ResourceInstanceRelease.find(releaseKey);

            if (release != m_ResourceInstanceRelease.end())
            {
                Dest.m_PRef.m_Pointer = URef.m_PRef.m_Pointer;
                ++release->second->m_RefCount;
            }
            else if (instance != m_ResourceInstance.end())
                Dest.m_PRef.m_Pointer = instance->second->m_pData;
        }

        /*!*****************************************************************************
            \brief Get the total number of resources managed by mgr
        *******************************************************************************/
        int getResourceCount()
        {
            return static_cast<int>( m_ResourceInstance.size() );
        }

    protected:
        /*!*****************************************************************************
            \brief Allocate the resource info

            \return Reference to an available instance_info
        *******************************************************************************/
        details::instance_info& AllocRscInfo( void )
        {
            details::instance_info& info = *m_pInfoBufferEmptyHead;
            m_pInfoBufferEmptyHead = reinterpret_cast<details::instance_info*>( m_pInfoBufferEmptyHead->m_pData );
            return info;
        }

        /*!*****************************************************************************
            \brief To release the Resource Info from the mgr

            \param [in] RscInfo: Data of the resource to be removed
        *******************************************************************************/
        void ReleaseRscInfo(details::instance_info& RscInfo)
        {
            uint64_t const releaseKey = reinterpret_cast<uint64_t>( RscInfo.m_pData );
            uint64_t const instanceKey = RscInfo.m_GUID.m_Value ^ RscInfo.m_TypeGUID.m_Value;

            m_ResourceInstance.erase(instanceKey);
            m_ResourceInstanceRelease.erase(releaseKey);
        }

        /*!*****************************************************************************
            \brief Fully allocate the instance info

            \param [in] pRsc: Address of the pointer of the resource to be added
            \param [in] RscGUID: The unique guid of the resource
            \param [in] TypeGUID: The guid of the type of resource
        *******************************************************************************/
        void FullInstanceInfoAlloc(void* pRsc, resource::guid RscGUID, resource::type_guid TypeGUID)
        {
            details::instance_info* pNext       = reinterpret_cast<details::instance_info*>(m_pInfoBufferEmptyHead->m_pData);
            m_pInfoBufferEmptyHead->m_pData     = pRsc;
            m_pInfoBufferEmptyHead->m_GUID      = RscGUID;
            m_pInfoBufferEmptyHead->m_TypeGUID  = TypeGUID;
            m_pInfoBufferEmptyHead->m_RefCount  = 1;

            uint64_t hashID = RscGUID.m_Value ^ TypeGUID.m_Value;
            m_ResourceInstance.insert({ hashID, m_pInfoBufferEmptyHead });
            m_ResourceInstanceRelease.insert({ reinterpret_cast<uint64_t>(pRsc), m_pInfoBufferEmptyHead });

            m_pInfoBufferEmptyHead = pNext;
        }

        /*!*****************************************************************************
            \brief Fully release the instance info

            \param [in] RscInfo: instance_info to be released
        *******************************************************************************/
        void FullInstanceInfoRelease( details::instance_info& RscInfo )
        {
            ReleaseRscInfo(RscInfo);
        }

        constexpr static auto max_resources_v = 1024;

        std::unordered_map<std::uint64_t, details::universal_type>  m_RegisteredTypes;
        std::unordered_map<std::uint64_t, details::instance_info*>  m_ResourceInstance;
        std::unordered_map<std::uint64_t, details::instance_info*>  m_ResourceInstanceRelease;
        details::instance_info*                                     m_pInfoBufferEmptyHead{ nullptr };
        std::array<details::instance_info, max_resources_v>         m_InfoBuffer;
    };
}

