FreeMarker Template example: ${message}  
 
=======================
===  County List   ====
=======================
<#list countries as country>
    ${country_index + 1}. ${country}
</#list>

<#function isThirdPartyEnabled>
    <#return (input!"") != "yes"/>
</#function>

    <#if !false>
    	hello
    </#if>

    <#if isThirdPartyEnabled()>
    	thridParty
    </#if>
